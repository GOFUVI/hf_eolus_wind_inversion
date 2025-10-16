#!/usr/bin/env python3
"""Merge Parquet files so there is exactly one file per partition directory.

The script is invoked by ``finalize_geoparquet.sh`` to collapse Athena CTAS
output—often many small part files—into a single object per partition. This
dramatically reduces the number of files we keep in S3, easing subsequent
cataloguing and improving Athena scan performance.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

import argparse
import os
import re
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq


def list_parquet_in_dir(path: str) -> List[str]:
    """Return all readable Parquet files within ``path``.

    Returns
    -------
    List[str]
        Sorted list of Parquet file paths suitable for merging.
    """
    files: List[str] = []
    for name in os.listdir(path):
        if name.startswith(".") or name.startswith("_") or name == "SUCCESS":
            continue
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        try:
            if os.path.getsize(p) == 0:
                continue
        except OSError:
            continue
        # Detect Parquet regardless of extension by probing the schema
        try:
            pq.read_schema(p)
        except Exception:
            continue
        files.append(p)
    files.sort()
    return files


def detect_output_filename(dirpath: str) -> str:
    """Derive a deterministic output filename based on the directory name.

    Returns
    -------
    str
        ``value.parquet`` when the directory name follows ``key=value`` or
        ``data.parquet`` otherwise.
    """
    base = os.path.basename(dirpath.rstrip("/"))
    m = re.match(r"[^=]*=(.*)", base)
    if m and m.group(1):
        value = m.group(1)
        return f"{value}.parquet"
    return "data.parquet"


def merge_files(files: List[str], out_path: str) -> None:
    """Merge multiple Parquet files into a single object.

    Parameters
    ----------
    files
        Parquet file paths to merge.
    out_path
        Destination Parquet file that will hold the merged content.
    """
    if len(files) == 1:
        # Single file: rename/move if needed
        src = files[0]
        if os.path.abspath(src) != os.path.abspath(out_path):
            os.replace(src, out_path)
        return

    # Multiple files: merge via ParquetWriter to avoid loading everything at once
    first = files[0]
    table0 = pq.read_table(first)
    schema = table0.schema
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        writer.write_table(table0)
        for f in files[1:]:
            t = pq.read_table(f)
            # Ensure compatible schema (allow metadata differences)
            if not t.schema.equals(schema, check_metadata=False):
                t = t.cast(schema)
            writer.write_table(t)

    # Remove originals
    for f in files:
        try:
            if os.path.abspath(f) != os.path.abspath(out_path):
                os.unlink(f)
        except OSError:
            pass


def is_leaf_dir_with_parquet(path: str) -> bool:
    """Return True when a directory contains Parquet payloads."""
    # A leaf directory is any directory that contains Parquet files, regardless of subfolders
    return len(list_parquet_in_dir(path)) > 0


def walk_partition_leaves(root: str) -> List[str]:
    """Collect directories that contain Parquet leaf files.

    Returns
    -------
    List[str]
        Directory paths that should each contain exactly one merged Parquet file
        after processing.
    """
    leaves: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if is_leaf_dir_with_parquet(dirpath):
            leaves.append(dirpath)
    # If no leaves found, check root itself for unpartitioned files
    if not leaves and is_leaf_dir_with_parquet(root):
        leaves.append(root)
    return leaves


def main():
    """CLI entry point for the merge utility."""
    ap = argparse.ArgumentParser(description="Merge Parquet files per partition directory")
    ap.add_argument("--root", required=True, help="Root directory with Parquet files")
    args = ap.parse_args()

    leaves = walk_partition_leaves(args.root)
    for d in leaves:
        files = list_parquet_in_dir(d)
        if not files:
            continue
        # Maintain a deterministic filename per partition to make downstream
        # cataloguing predictable.
        out_name = detect_output_filename(d)
        out_path = os.path.join(d, out_name)
        merge_files(files, out_path)


if __name__ == "__main__":
    main()
