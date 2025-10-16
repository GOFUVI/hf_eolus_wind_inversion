#!/usr/bin/env python3
"""Add GeoParquet metadata to Parquet files.

The utility equips Parquet datasets with the metadata mandated by the
GeoParquet 1.0 specification. It is invoked locally by the
``finalize_geoparquet.sh`` pipeline, but it can also be executed directly
against either a local directory tree or an S3 prefix. The script inspects the
geometry column—or columns, if reused across partitions—to determine geometry
encoding (WKB versus WKT), detects the geometry types present, computes a global
bounding box, and patches every Parquet file in place via atomic writes.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import argparse
import json
import os
import tempfile
from typing import Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import shapely.wkb as wkb
import shapely.wkt as wkt
from shapely.ops import unary_union
import math

CRS84 = {
    "$schema": "https://proj.org/schemas/v0.5/projjson.schema.json",
    "type": "GeographicCRS",
    "name": "WGS 84 longitude-latitude",
    "datum": {
        "type": "GeodeticReferenceFrame",
        "name": "World Geodetic System 1984",
        "ellipsoid": {
            "name": "WGS 84",
            "semi_major_axis": 6378137,
            "inverse_flattening": 298.257223563,
        },
    },
    "coordinate_system": {
        "subtype": "ellipsoidal",
        "axis": [
            {
                "name": "Geodetic longitude",
                "abbreviation": "Lon",
                "direction": "east",
                "unit": "degree",
            },
            {
                "name": "Geodetic latitude",
                "abbreviation": "Lat",
                "direction": "north",
                "unit": "degree",
            },
        ],
    },
    "id": {"authority": "OGC", "code": "CRS84"},
}


def parse_s3(uri: str) -> Tuple[str, str]:
    """Split an S3 URI into bucket and prefix components.

    Returns
    -------
    Tuple[str, str]
        ``(bucket, prefix)`` derived from the provided URI. The prefix always
        ends with a trailing slash to simplify subsequent key joins.
    """
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    bucket_key = uri[5:]
    parts = bucket_key.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) == 2 else ""
    return bucket, prefix.rstrip("/") + "/"


def list_parquet_objects(s3, bucket: str, prefix: str) -> Iterable[str]:
    """Yield candidate Parquet object keys under a prefix, skipping markers.

    The generator excludes folders, Athena scratch outputs, and other
    non-Parquet artefacts so downstream processing can iterate confidently.
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip S3 pseudo-directories and Athena result folders
            if key.endswith("/") or "/query_results/" in key:
                continue
            # Do not rely on extension; downstream will validate
            yield key


def compute_bbox_s3(
    files: Iterable[str], s3, bucket, geometry_column: str, encoding: str
) -> List[float]:
    """Compute dataset-wide bounds by sampling geometries from S3 objects.

    Returns
    -------
    List[float]
        Bounding box formatted as ``[minx, miny, maxx, maxy]`` in CRS84. When
        no valid geometries are found, the fallback is a degenerate zero-valued
        bbox so metadata consumers still receive a valid array.
    """
    has_bounds = False
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for key in files:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            s3.download_file(bucket, key, tmp.name)
            table = pq.read_table(tmp.name, columns=[geometry_column])
            for g in table.column(geometry_column).to_pylist():
                if g is None:
                    continue
                try:
                    geom = wkt.loads(g) if encoding.upper() == "WKT" else wkb.loads(g)
                except Exception:
                    continue
                bx = geom.bounds  # (minx, miny, maxx, maxy)
                if not (len(bx) == 4 and all(isinstance(v, (int, float)) for v in bx)):
                    continue
                if not all(math.isfinite(v) for v in bx):
                    continue
                has_bounds = True
                minx = min(minx, bx[0])
                miny = min(miny, bx[1])
                maxx = max(maxx, bx[2])
                maxy = max(maxy, bx[3])
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    if not has_bounds:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(minx), float(miny), float(maxx), float(maxy)]


def patch_file_s3(
    key: str,
    bbox: List[float],
    s3,
    bucket: str,
    geometry_column: str,
    geometry_types: List[str],
    encoding: str,
) -> bool:
    """Write GeoParquet metadata into an object stored on S3.

    Returns
    -------
    bool
        ``True`` when the object is patched successfully, ``False`` otherwise.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        s3.download_file(bucket, key, tmp.name)
        table = pq.read_table(tmp.name)
        meta = dict(table.schema.metadata or {})
        geo = {
            "version": "1.0.0",
            "primary_column": geometry_column,
            "columns": {
                geometry_column: {
                    "encoding": encoding,
                    "geometry_types": geometry_types,
                    "crs": CRS84,
                    "bbox": bbox,
                }
            },
        }
        meta[b"geo"] = json.dumps(geo).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, tmp.name)
        s3.upload_file(tmp.name, bucket, key)
        return True
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def list_parquet_local(root_dir: str) -> List[str]:
    """Collect valid Parquet files from a local directory tree.

    Returns
    -------
    List[str]
        Sorted list of file paths pointing to Parquet payloads.
    """
    files_out: List[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for f in files:
            # Skip auxiliary files and hidden markers
            if f.startswith('.') or f.startswith('_') or f == 'SUCCESS':
                continue
            p = os.path.join(base, f)
            # Skip empty files
            try:
                if os.path.getsize(p) == 0:
                    continue
            except OSError:
                continue
            # Validate Parquet by probing schema (filters out stray files)
            try:
                pq.read_schema(p)
            except Exception:
                continue
            files_out.append(p)
    files_out.sort()
    return files_out


def compute_bbox_local(files: Iterable[str], geometry_column: str, encoding: str) -> List[float]:
    """Compute dataset-wide bounds using local Parquet files.

    Returns
    -------
    List[float]
        Bounding box formatted as ``[minx, miny, maxx, maxy]`` in CRS84, or a
        zero-valued fallback when no valid geometries are available.
    """
    has_bounds = False
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for path in files:
        try:
            table = pq.read_table(path, columns=[geometry_column])
            for g in table.column(geometry_column).to_pylist():
                if g is None:
                    continue
                try:
                    geom = wkt.loads(g) if encoding.upper() == "WKT" else wkb.loads(g)
                except Exception:
                    continue
                bx = geom.bounds  # (minx, miny, maxx, maxy)
                if not (len(bx) == 4 and all(isinstance(v, (int, float)) for v in bx)):
                    continue
                if not all(math.isfinite(v) for v in bx):
                    continue
                has_bounds = True
                minx = min(minx, bx[0])
                miny = min(miny, bx[1])
                maxx = max(maxx, bx[2])
                maxy = max(maxy, bx[3])
        except Exception:
            continue
    if not has_bounds:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(minx), float(miny), float(maxx), float(maxy)]


def patch_file_local(
    path: str,
    bbox: List[float],
    geometry_column: str,
    geometry_types: List[str],
    encoding: str,
) -> bool:
    """Inject GeoParquet metadata into a local Parquet file atomically.

    Returns
    -------
    bool
        ``True`` if the file was successfully rewritten, ``False`` otherwise.
    """
    try:
        table = pq.read_table(path)
    except Exception as e:
        print(f"PATCH_ERROR: read_table failed for {path}: {e}")
        return False
    # Detect if geometry column exists; still write metadata even if missing to aid debugging
    schema: pa.Schema = table.schema
    has_geom = geometry_column in schema.names
    if not has_geom:
        print(f"WARNING: geometry column '{geometry_column}' not found in {path}. Writing geo metadata anyway.")
    meta = dict(schema.metadata or {})
    geo = {
        "version": "1.0.0",
        "primary_column": geometry_column,
        "columns": {
            geometry_column: {
                "encoding": encoding,
                "geometry_types": geometry_types,
                "crs": CRS84,
                "bbox": bbox,
            }
        },
    }
    try:
        meta[b"geo"] = json.dumps(geo).encode("utf-8")
    except Exception as e:
        print(f"PATCH_ERROR: failed to serialize geo metadata for {path}: {e}")
        return False
    table = table.replace_schema_metadata(meta)
    # Write to a temp file then replace to be atomic and avoid partial writes
    tmp_out = None
    try:
        d = os.path.dirname(path)
        fd, tmp_out = tempfile.mkstemp(prefix=".geo_meta_", suffix=".parquet", dir=d)
        os.close(fd)
        pq.write_table(table, tmp_out)
        os.replace(tmp_out, path)
        return True
    except Exception as e:
        print(f"PATCH_ERROR: failed to write metadata for {path}: {e}")
        try:
            if tmp_out and os.path.exists(tmp_out):
                os.unlink(tmp_out)
        except Exception:
            pass
        return False


def detect_geometry_types_from_sample(
    files: Iterable[str], geometry_column: str, encoding: str, max_geoms: int = 200
) -> List[str]:
    """Infer geometry types present in the dataset using shallow sampling.

    Returns
    -------
    List[str]
        Sorted list of geometry type names compatible with GeoParquet.
    """
    seen = set()
    count = 0
    for path in files:
        try:
            table = pq.read_table(path, columns=[geometry_column])
            for g in table.column(geometry_column).to_pylist():
                if g is None:
                    continue
                try:
                    shp = wkt.loads(g) if encoding.upper() == "WKT" else wkb.loads(g)
                except Exception:
                    continue
                t = shp.geom_type
                # Map shapely names to GeoParquet names (they match)
                seen.add(t)
                count += 1
                if count >= max_geoms:
                    break
        except Exception:
            continue
        if count >= max_geoms:
            break
    if not seen:
        # Fallback to broadest type
        return ["Geometry"]
    # Normalize Multi vs single types present
    return sorted(seen)


def detect_geometry_encoding(files: Iterable[str], geometry_column: str) -> str:
    """Infer geometry encoding from Arrow schema and a small data sample.

    Returns
    -------
    str
        Either ``"WKB"`` or ``"WKT"``, defaulting to ``"WKB"`` when the data
        does not provide enough evidence.
    """
    # Try schema-based detection on first readable file
    for path in files:
        try:
            schema = pq.read_schema(path)
            if geometry_column in schema.names:
                t = schema.field(geometry_column).type
                if pa.types.is_binary(t) or pa.types.is_large_binary(t):
                    return "WKB"
                if pa.types.is_string(t) or pa.types.is_large_string(t):
                    return "WKT"
        except Exception:
            continue
    # Fallback to sampling data
    for path in files:
        try:
            table = pq.read_table(path, columns=[geometry_column])
            col = table.column(geometry_column)
            values = col.to_pylist()[:5]
            for v in values:
                if v is None:
                    continue
                if isinstance(v, (bytes, bytearray, memoryview)):
                    return "WKB"
                if isinstance(v, str):
                    return "WKT"
        except Exception:
            continue
    return "WKB"


def main():
    """CLI entry point handling both local and S3 execution modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-path", required=False, help="s3://bucket/prefix of Parquet dataset")
    parser.add_argument("--local-path", required=False, help="Local directory containing Parquet files")
    parser.add_argument("--geometry-column", required=False, default="geometry", help="Name of the geometry column (default: geometry)")
    args = parser.parse_args()

    if bool(args.s3_path) == bool(args.local_path):
        raise SystemExit("Provide exactly one of --s3-path or --local-path")

    if args.local_path:
        files = list_parquet_local(args.local_path)
        if not files:
            print("No Parquet files detected for local path; nothing to do.")
            return
        # Compute shared metadata once; these values feed every subsequent
        # patching call so we avoid recomputing heavy geometry operations.
        encoding = detect_geometry_encoding(files, args.geometry_column)
        geometry_types = detect_geometry_types_from_sample(files, args.geometry_column, encoding)
        bbox = compute_bbox_local(files, args.geometry_column, encoding)
        patched = 0
        attempted = 0
        for path in files:
            attempted += 1
            if patch_file_local(path, bbox, args.geometry_column, geometry_types, encoding):
                patched += 1
        # Simple verification on first parquet-able file (schema-only)
        sample_meta_printed = False
        errors = []
        for path in files:
            try:
                schema = pq.read_schema(path)
                meta = schema.metadata or {}
                print(f"Patched {patched}/{attempted} files. Sample: {path}")
                print("HAS_GEO=", b"geo" in meta)
                if b"geo" in meta:
                    try:
                        print("GEO_JSON=", (meta[b"geo"].decode())[:400])
                    except Exception:
                        pass
                sample_meta_printed = True
                break
            except Exception as e:
                errors.append(f"{path}: {e}")
                continue
        if not sample_meta_printed:
            print(f"Patched {patched}/{attempted} files. No readable Parquet files found for verification.")
            for msg in errors[:5]:
                print("VERIFY_ERROR:", msg)
    else:
        # Lazy import to avoid requiring boto3 when not needed
        import boto3  # type: ignore

        bucket, prefix = parse_s3(args.s3_path)
        s3 = boto3.client("s3")
        files = list(list_parquet_objects(s3, bucket, prefix))
        # Download a handful of objects locally so that the schema/geometry
        # detection routines can reuse the exact same logic as the local branch
        # without iterating over every S3 object.
        tmpdir = tempfile.TemporaryDirectory()
        sample_paths = []
        for i, key in enumerate(files):
            if i >= 5:
                break
            p = os.path.join(tmpdir.name, f"sample_{i}.parquet")
            try:
                s3.download_file(bucket, key, p)
                sample_paths.append(p)
            except Exception:
                continue
        encoding = detect_geometry_encoding(sample_paths, args.geometry_column)
        geometry_types = detect_geometry_types_from_sample(sample_paths, args.geometry_column, encoding)
        bbox = compute_bbox_s3(files, s3, bucket, args.geometry_column, encoding)
        attempted = 0
        patched = 0
        for key in files:
            attempted += 1
            if patch_file_s3(key, bbox, s3, bucket, args.geometry_column, geometry_types, encoding):
                patched += 1
        print(f"Patched {patched}/{attempted} S3 objects under s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main()
