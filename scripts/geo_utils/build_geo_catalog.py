#!/usr/bin/env python3
"""Build a self-contained STAC catalog for GeoParquet assets.

Features:
- Moves/copies GeoParquet assets into an `assets/` directory under the catalog root.
- Emits one STAC Item per asset into `items/` (mirrors partition subfolders if any).
- Writes a root STAC Collection at `collection.json`.
- Creates nested sub-catalogs inside the Collection for partition directories
  (e.g., `items/pos_bragg=1/catalog.json`).
- Uses only relative HREFs so the catalog is portable (offline/distributable).
- Includes the STAC CLI options present in the reference SAR example
  (`--collection-id`, `--item-properties`, `--collection-properties`).

Notes:
- Partition detection is path-based: any directory segment matching `name=value`
  is treated as a partition level.
- Geometry and spatial extent are derived from GeoParquet metadata (`geo` key)
  when available; otherwise, a best-effort scan is attempted.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pystac
from pystac.extensions.table import Column, Table, TableExtension
from shapely.geometry import box, mapping
from pystac.layout import TemplateLayoutStrategy


# -------------------------------
# Utilities and type conversions
# -------------------------------


def to_aware_utc(dt: datetime) -> datetime:
    """Convert naive datetimes to UTC and normalise aware ones to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def date_to_datetime_utc(d: date) -> datetime:
    """Lift a date into a midnight UTC datetime."""
    return datetime.combine(d, time(0, 0, 0)).replace(tzinfo=timezone.utc)


def parse_str_datetime(value: str) -> Optional[datetime]:
    """Parse diverse ISO-8601 string encodings into aware UTC datetimes."""
    v = value.strip()
    # Try common ISO-8601 forms
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            if fmt == "%Y-%m-%d":
                return date_to_datetime_utc(datetime.strptime(v, fmt).date())
            dt = datetime.strptime(v, fmt)
            return to_aware_utc(dt)
        except Exception:
            continue
    return None


def to_rfc3339_z(dt: datetime) -> str:
    """Render the datetime using RFC-3339 with an explicit Z suffix."""
    dt = to_aware_utc(dt)
    s = dt.isoformat()
    return s[:-6] + "Z" if s.endswith("+00:00") else (s if s.endswith("Z") else s + "Z")


def is_partition_segment(seg: str) -> bool:
    """Detect Hive-style partition directory segments like 'key=value'."""
    return bool(re.match(r"^[^=]+=.+$", seg))


def walk_parquet_files(root: Path) -> Iterable[Path]:
    """Yield all valid Parquet payloads under a root, ignoring metadata files."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(".") or name.startswith("_") or name == "SUCCESS":
            continue
        # Probe parquet-ness by trying to read the schema
        try:
            pq.read_schema(p)
        except Exception:
            continue
        yield p


def arrow_type_to_table_type(t: pa.DataType) -> str:
    """Map Arrow data types to STAC Table extension type identifiers."""
    if pa.types.is_integer(t):
        return "integer"
    if pa.types.is_floating(t):
        return "number"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_timestamp(t):
        return "datetime"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "binary"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    # Fallback: serialize as string
    return "string"


def columns_from_schema(schema: pa.Schema) -> List[Dict[str, Any]]:
    """Translate a PyArrow schema into STAC column descriptors."""
    cols: List[Dict[str, Any]] = []
    for f in schema:
        cols.append(
            {
                "name": f.name,
                "type": arrow_type_to_table_type(f.type),
            }
        )
    return cols


@dataclass
class GeoMeta:
    primary_column: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]


def geometa_from_parquet(p: Path) -> GeoMeta:
    """Inspect GeoParquet metadata and extract primary geometry + bbox."""
    try:
        schema = pq.read_schema(p)
        meta = schema.metadata or {}
        if b"geo" in meta:
            try:
                gj = json.loads(meta[b"geo"].decode("utf-8"))
                primary = gj.get("primary_column")
                bbox = None
                if primary:
                    bbox = gj.get("columns", {}).get(primary, {}).get("bbox")
                    if bbox and len(bbox) == 4:
                        return GeoMeta(primary, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                # If no primary or bbox at column-level, try top-level bbox
                top_bbox = gj.get("bbox")
                if top_bbox and len(top_bbox) == 4:
                    return GeoMeta(primary, (float(top_bbox[0]), float(top_bbox[1]), float(top_bbox[2]), float(top_bbox[3])))
                return GeoMeta(primary, None)
            except Exception:
                return GeoMeta(None, None)
        return GeoMeta(None, None)
    except Exception:
        return GeoMeta(None, None)


def extract_times(p: Path, candidate_cols: List[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Attempt to compute [start, end] observation times from candidate columns.

    Strategy:
    - If a pair like (firstMeasurementTime, lastMeasurementTime) exists, use min(first), max(last)
    - Else, for any single time-like column found, use min(col), max(col)
    - Supports Arrow date, timestamp, or ISO-like string columns
    """
    schema = pq.read_schema(p)
    present = [c for c in candidate_cols if c in schema.names]
    if not present:
        return (None, None)
    # Read only present columns. In some datasets, non-selected columns can
    # have inconsistent encodings across row groups/files (e.g., int32 vs
    # dictionary-encoded int32). The dataset engine may try to unify schemas
    # and fail even if those columns aren't requested. Fall back to reading
    # via ParquetFile to avoid dataset-wide schema unification.
    try:
        table = pq.read_table(p, columns=present)
    except pa.ArrowTypeError:
        # Fallback path avoids merging unused columns like 'pos_bragg'
        table = pq.ParquetFile(p).read(columns=present)

    def normalize_val(v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return to_aware_utc(v)
        if isinstance(v, date):
            return date_to_datetime_utc(v)
        if isinstance(v, str):
            return parse_str_datetime(v)
        return None

    # Preferred paired names
    pairs = [("firstMeasurementTime", "lastMeasurementTime")]
    for a, b in pairs:
        if a in table.column_names and b in table.column_names:
            first_vals = [normalize_val(x) for x in table[a].to_pylist()]
            last_vals = [normalize_val(x) for x in table[b].to_pylist()]
            first_vals = [x for x in first_vals if x is not None]
            last_vals = [x for x in last_vals if x is not None]
            if first_vals and last_vals:
                return (min(first_vals), max(last_vals))

    # Fallback: use any single column's min/max
    for c in present:
        vals = [normalize_val(x) for x in table[c].to_pylist()]
        vals = [x for x in vals if x is not None]
        if vals:
            return (min(vals), max(vals))
    return (None, None)


def ensure_dir(p: Path) -> None:
    """Create a directory path idempotently."""
    p.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    """Move or copy a file while ensuring the destination hierarchy exists."""
    ensure_dir(dst.parent)
    if copy:
        # shutil.copy2 keeps metadata but avoid bringing as extra import; os.link fallback if possible
        try:
            import shutil

            shutil.copy2(src, dst)
        except Exception:
            # Fallback to simple copy
            with open(src, "rb") as r, open(dst, "wb") as w:
                w.write(r.read())
    else:
        os.replace(src, dst)


def relpath(from_dir: Path, to_path: Path) -> str:
    """Return a relative path using the host OS semantics."""
    return os.path.relpath(to_path, start=from_dir)


# -------------------------------
# Core catalog build
# -------------------------------


@dataclass
class AssetInfo:
    parquet_path: Path
    rel_within_assets: Path  # relative to assets_dir
    partitions: List[str]  # e.g., ["date=2024-01-01", "pos_bragg=1"]
    bbox: Optional[Tuple[float, float, float, float]]
    start: Optional[datetime]
    end: Optional[datetime]
    row_count: int
    schema: pa.Schema
    primary_geom: Optional[str]


def collect_assets(assets_dir: Path) -> List[AssetInfo]:
    """Enumerate every GeoParquet asset staged under `assets/` and gather metadata."""
    infos: List[AssetInfo] = []
    for p in walk_parquet_files(assets_dir):
        rel = p.relative_to(assets_dir)
        # Preserve partition directory levels so they can be reconstructed as subcatalogs.
        parts = [seg for seg in rel.parts[:-1] if is_partition_segment(seg)]
        geo = geometa_from_parquet(p)
        # Prefer bbox from metadata; leave None if absent
        bbox = geo.bbox
        # Time extraction: try common candidates
        start, end = extract_times(
            p,
            [
                "firstMeasurementTime",
                "lastMeasurementTime",
                "start_datetime",
                "end_datetime",
                "timestamp",
                "date",
                "time",
            ],
        )
        pf = pq.ParquetFile(p)
        row_count = pf.metadata.num_rows if pf.metadata is not None else 0
        schema = pf.schema_arrow
        infos.append(
            AssetInfo(
                parquet_path=p,
                rel_within_assets=rel,
                partitions=parts,
                bbox=bbox,
                start=start,
                end=end,
                row_count=row_count,
                schema=schema,
                primary_geom=geo.primary_column,
            )
        )
    infos.sort(key=lambda x: str(x.rel_within_assets))
    # Sorting stabilises downstream materialisation order for reproducible catalogs.
    return infos


class PartitionNode:
    def __init__(self, name: str | None = None):
        """Lightweight trie that mirrors the directory partition hierarchy."""
        self.name = name  # e.g., "date=2024-01-01"
        self.children: Dict[str, PartitionNode] = {}
        self.items: List[AssetInfo] = []

    def add(self, parts: List[str], info: AssetInfo) -> None:
        """Attach an asset to the node determined by its partition segments."""
        if not parts:
            self.items.append(info)
            return
        head, *tail = parts
        if head not in self.children:
            self.children[head] = PartitionNode(name=head)
        self.children[head].add(tail, info)


def build_partition_tree(infos: List[AssetInfo]) -> PartitionNode:
    """Construct a partition tree covering all assets for nested catalog creation."""
    root = PartitionNode()
    for info in infos:
        root.add(info.partitions, info)
    return root


def aggregate_overall_bbox_and_time(infos: List[AssetInfo]) -> Tuple[Optional[List[float]], Optional[datetime], Optional[datetime]]:
    """Aggregate spatial/temporal extents across every asset."""
    bbox: Optional[List[float]] = None
    starts: List[datetime] = []
    ends: List[datetime] = []
    for i in infos:
        if i.bbox is not None:
            if bbox is None:
                bbox = list(i.bbox)
            else:
                bbox = [
                    min(bbox[0], i.bbox[0]),
                    min(bbox[1], i.bbox[1]),
                    max(bbox[2], i.bbox[2]),
                    max(bbox[3], i.bbox[3]),
                ]
        if i.start is not None:
            starts.append(i.start)
        if i.end is not None:
            ends.append(i.end)
    overall_start = min(starts) if starts else None
    overall_end = max(ends) if ends else None
    # Returning None allows STAC to elide the extent when no timestamps are available.
    return bbox, overall_start, overall_end


def item_id_for(info: AssetInfo) -> str:
    """Deterministically derive an item identifier from the asset path."""
    base = "_".join([*info.rel_within_assets.parts[:-1], info.rel_within_assets.stem]) or info.rel_within_assets.stem
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", base)


def make_item_for_asset(info: AssetInfo, items_dir: Path, assets_dir: Path, item_self_dir: Path, item_props: Dict[str, Any]) -> pystac.Item:
    """Create a STAC Item describing a single GeoParquet asset."""
    # Geometry: if bbox available, use bbox polygon; otherwise leave geometry None (invalid per STAC, but unlikely)
    if info.bbox is not None:
        geom = mapping(box(*info.bbox))
        bbox = list(info.bbox)
    else:
        geom = None
        bbox = None

    # Choose item datetime/start/end
    props = dict(item_props)
    if info.start is not None:
        props["start_datetime"] = to_rfc3339_z(info.start)
    if info.end is not None:
        props["end_datetime"] = to_rfc3339_z(info.end)

    # STAC requires either datetime OR start/end; set datetime to start if present
    dt_for_item: Optional[datetime] = info.start

    item_id = item_id_for(info)

    # Instantiate the core item with geometry, temporal range, and custom properties.
    it = pystac.Item(
        id=item_id,
        geometry=geom,
        bbox=bbox,
        datetime=dt_for_item,
        properties=props,
    )

    # Compute correct relative href from the item directory to the asset file
    asset_abs = assets_dir / info.rel_within_assets
    rel_href_from_item = relpath(item_self_dir, asset_abs)
    it.add_asset(
        "data",
        pystac.Asset(href=rel_href_from_item, media_type=pystac.MediaType.PARQUET, roles=["data"]),
    )

    # Table extension with columns and row_count
    # Build as individual fields and allow overrides from item_props
    TableExtension.add_to(it)
    it_ext = TableExtension.ext(it)

    # Columns: allow override via properties["table:columns"], else infer from schema
    cols_override = props.get("table:columns")
    if cols_override is not None:
        it_ext.columns = [Column(c) for c in cols_override]
    else:
        inferred_cols = columns_from_schema(info.schema)
        it_ext.columns = [Column(c) for c in inferred_cols]

    # Row count: allow override via properties["table:row_count"], else use computed
    row_count_override = props.get("table:row_count")
    if row_count_override is not None:
        it_ext.row_count = int(row_count_override)
    else:
        it_ext.row_count = info.row_count

    # Primary geometry: allow override via properties["table:primary_geometry"], else use detected
    primary_geom_override = props.get("table:primary_geometry")
    if primary_geom_override is not None:
        it_ext.primary_geometry = str(primary_geom_override)
    elif info.primary_geom:
        it_ext.primary_geometry = info.primary_geom

    return it


def build_and_save_catalog(
    root_dir: Path,
    collection_id: str,
    item_props_path: Optional[str],
    collection_props_path: Optional[str],
    move_from: Optional[Path] = None,
    copy_instead_of_move: bool = False,
) -> None:
    """Materialise a partition-aware STAC catalog rooted at `root_dir`."""
    root_dir = root_dir.resolve()
    assets_dir = root_dir / "assets"
    items_root = root_dir / "items"
    ensure_dir(assets_dir)
    ensure_dir(items_root)

    # If move_from provided, move/copy parquet assets under assets_dir preserving partitions
    if move_from is not None:
        move_from = move_from.resolve()
        for p in walk_parquet_files(move_from):
            rel = p.relative_to(move_from)
            dst = assets_dir / rel
            move_or_copy(p, dst, copy=copy_instead_of_move)

    # Collect assets under assets_dir
    infos = collect_assets(assets_dir)
    if not infos:
        raise SystemExit(f"No Parquet assets found under {assets_dir}")

    # Load extra properties
    item_props: Dict[str, Any] = {}
    collection_props: Dict[str, Any] = {}
    if item_props_path:
        with open(item_props_path, "r", encoding="utf-8") as f:
            item_props = json.load(f)
    if collection_props_path:
        with open(collection_props_path, "r", encoding="utf-8") as f:
            collection_props = json.load(f)

    # Root collection extents
    overall_bbox, overall_start, overall_end = aggregate_overall_bbox_and_time(infos)
    spatial_extent = pystac.SpatialExtent([overall_bbox if overall_bbox else [0, 0, 0, 0]])
    temporal_extent = pystac.TemporalExtent([[overall_start, overall_end]])

    coll_license = collection_props.pop("license", "GPL-3.0")
    collection = pystac.Collection(
        id=collection_id,
        description=collection_props.pop(
            "description",
            "GeoParquet assets produced by hf_eolus_geo_tools aggregation.",
        ),
        extent=pystac.Extent(spatial_extent, temporal_extent),
        license=coll_license,
        extra_fields=collection_props,
    )
    # Apply the Table extension at collection level as individual fields, not a single object
    # This enables overriding specific elements (e.g., columns) via --collection-properties JSON
    TableExtension.add_to(collection)

    # Defaults derived from assets
    total_rows = sum(i.row_count for i in infos)
    default_columns = columns_from_schema(infos[0].schema)

    # Respect user-provided overrides if present in collection.extra_fields
    coll_extra = collection.extra_fields

    # table:tables must be an array of simple objects (only name and optional description)
    # Provide a minimal default if not supplied by the user
    if "table:tables" not in coll_extra:
        coll_extra["table:tables"] = [
            {
                "name": collection_id,
                "description": coll_extra.get(
                    "table:description",
                    "Aggregated GeoParquet table",
                ),
            }
        ]

    # Set columns and row_count as top-level table:* fields if not provided
    if "table:columns" not in coll_extra:
        coll_extra["table:columns"] = default_columns
    if "table:row_count" not in coll_extra:
        coll_extra["table:row_count"] = int(total_rows)

    # Partitioned layout
    tree = build_partition_tree(infos)

    # Prepare absolute self-hrefs for root and children to ensure relative linking is correct
    collection.set_self_href(str(root_dir / "collection.json"))
    parent_catalog_path = root_dir.parent / "catalog.json"
    parent_catalog: Optional[pystac.Catalog] = None
    if parent_catalog_path.is_file():
        try:
            parent_catalog = pystac.Catalog.from_file(str(parent_catalog_path))
            parent_catalog.set_self_href(str(parent_catalog_path))
        except Exception:
            parent_catalog = None

    def materialize(node: PartitionNode, parent: pystac.Catalog, rel_parts: List[str]) -> None:
        """Populate STAC items/catalogs corresponding to a partition tree node."""
        # If node has items and no partitions, items go directly under parent
        if not node.children:
            # Always write items under items/ root for portability and simplicity
            ensure_dir(items_root)
            for i in node.items:
                item_self_dir = items_root
                fname = f"{item_id_for(i)}.json"
                item_path = item_self_dir / fname
                it = make_item_for_asset(i, items_root, assets_dir, item_self_dir, item_props)
                it.set_self_href(str(item_path))
                parent.add_item(it)
            return

        # Otherwise create subcatalogs for each child and recurse
        for seg, child in sorted(node.children.items(), key=lambda kv: kv[0]):
            sub_rel = rel_parts + [seg]
            sub_dir = items_root.joinpath(*sub_rel)
            ensure_dir(sub_dir)
            subcat = pystac.Catalog(id=re.sub(r"[^A-Za-z0-9_\-]+", "_", seg), description=f"Partition {seg}")
            subcat.set_self_href(str(sub_dir / "catalog.json"))
            parent.add_child(subcat)
            # Add items at this level (if any) before recursing further
            if child.items:
                for i in child.items:
                    # Always write items under items/ root
                    item_self_dir = items_root
                    fname = f"{item_id_for(i)}.json"
                    item_path = item_self_dir / fname
                    it = make_item_for_asset(i, items_root, assets_dir, item_self_dir, item_props)
                    it.set_self_href(str(item_path))
                    subcat.add_item(it)
            if child.children:
                materialize(child, subcat, sub_rel)

    materialize(tree, collection, [])

    # Normalize HREFs so that:
    # - collection.json sits at root
    # - items live under items/${id}.json
    # - subcatalogs (partitions) live under items/${id}/catalog.json
    layout = TemplateLayoutStrategy(
        catalog_template="items/${id}/catalog.json",
        collection_template="collection.json",
        item_template="items/${id}.json",
    )
    collection.normalize_hrefs(str(root_dir), strategy=layout)

    if parent_catalog is not None:
        parent_href_rel = Path(relpath(root_dir, parent_catalog_path)).as_posix()
        collection.remove_links(pystac.RelType.PARENT)
        collection.remove_links(pystac.RelType.ROOT)
        collection.add_link(
            pystac.Link(rel=pystac.RelType.PARENT, target=parent_href_rel, media_type=pystac.MediaType.JSON)
        )
        collection.add_link(
            pystac.Link(rel=pystac.RelType.ROOT, target=parent_href_rel, media_type=pystac.MediaType.JSON)
        )

    # Save as self-contained (relative links)
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    if parent_catalog is not None:
        target_collection_path = (root_dir / "collection.json").resolve()
        parent_dir = parent_catalog_path.parent
        desired_child_href = Path(relpath(parent_dir, target_collection_path)).as_posix()
        child_exists = False
        for link in parent_catalog.links:
            if link.rel != pystac.RelType.CHILD:
                continue
            href = link.get_href(parent_catalog)
            if href and Path(href).resolve() == target_collection_path:
                child_exists = True
                # Harmonize href to the desired relative form if needed
                if (link.href or "") != desired_child_href:
                    link.href = desired_child_href
                    link.target = desired_child_href
                    link.set_owner(parent_catalog)
                break
        if not child_exists:
            new_link = pystac.Link(
                rel=pystac.RelType.CHILD,
                target=desired_child_href,
                media_type=pystac.MediaType.JSON,
                title=collection.extra_fields.get("title") or collection.id,
            )
            new_link.set_owner(parent_catalog)
            parent_catalog.add_link(new_link)
        parent_catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)


def main() -> None:
    """CLI entry point orchestrating argument parsing and catalog build."""
    ap = argparse.ArgumentParser(description="Build a STAC catalog for GeoParquet assets (partition-aware, self-contained)")
    ap.add_argument("root", help="Catalog root directory (collection.json will be placed here)")
    ap.add_argument("--collection-id", required=True, dest="collection_id", help="STAC collection identifier")
    ap.add_argument("--item-properties", dest="item_props", help="JSON file with extra STAC item properties")
    ap.add_argument("--collection-properties", dest="collection_props", help="JSON file with extra STAC collection properties")
    ap.add_argument(
        "--source-dir",
        dest="source_dir",
        help="Optional source directory with Parquet files to move into assets/ (preserving structure)",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy assets instead of moving them into assets/",
    )
    args = ap.parse_args()

    root = Path(args.root)
    move_from = Path(args.source_dir) if args.source_dir else None
    build_and_save_catalog(
        root_dir=root,
        collection_id=args.collection_id,
        item_props_path=args.item_props,
        collection_props_path=args.collection_props,
        move_from=move_from,
        copy_instead_of_move=bool(args.copy),
    )


if __name__ == "__main__":
    main()
