#!/usr/bin/env python3
"""Generate AWS Glue table input JSON for a Parquet dataset.

This helper is executed from ``finalize_geoparquet.sh`` once the GeoParquet
post-processing stage completes. The script reads the dataset schema using
PyArrow, maps each Arrow type to its Glue/Hive counterpart, and emits a fully
formed ``create-table`` payload that can be fed directly to ``aws glue`` or any
upstream orchestration layer. Partition columns are handled explicitly so that
Glue understands both their schema and the Hive-style layout of the files.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.types as patypes


def _arrow_type_to_glue(pa_type: pa.DataType, warnings: List[str]) -> str:
    """Map a PyArrow data type to the closest Glue/Hive primitive.

    Parameters
    ----------
    pa_type
        The Arrow type to translate.
    warnings
        Collector list that captures lossy conversions so the caller can surface
        them to the user after the schema walk finishes.

    Returns
    -------
    str
        Glue/Hive type compatible with the provided Arrow type.
    """
    if patypes.is_dictionary(pa_type):
        return _arrow_type_to_glue(pa_type.value_type, warnings)
    if patypes.is_string(pa_type) or patypes.is_large_string(pa_type):
        return "string"
    if patypes.is_binary(pa_type) or patypes.is_large_binary(pa_type) or patypes.is_fixed_size_binary(pa_type):
        return "binary"
    if patypes.is_boolean(pa_type):
        return "boolean"
    if any(
        checker(pa_type)
        for checker in (
            patypes.is_int8,
            patypes.is_int16,
            patypes.is_int32,
            patypes.is_uint8,
            patypes.is_uint16,
            patypes.is_uint32,
        )
    ):
        return "int"
    if patypes.is_int64(pa_type) or patypes.is_uint64(pa_type):
        return "bigint"
    if patypes.is_float16(pa_type) or patypes.is_float32(pa_type):
        return "float"
    if patypes.is_float64(pa_type):
        return "double"
    if patypes.is_timestamp(pa_type):
        return "timestamp"
    if patypes.is_date(pa_type):
        return "date"
    if patypes.is_time(pa_type):
        return "int"
    if patypes.is_duration(pa_type):
        return "bigint"
    if patypes.is_decimal(pa_type):
        return f"decimal({pa_type.precision},{pa_type.scale})"
    if patypes.is_list(pa_type) or patypes.is_large_list(pa_type) or patypes.is_fixed_size_list(pa_type):
        value_type = _arrow_type_to_glue(pa_type.value_type, warnings)
        return f"array<{value_type}>"
    if patypes.is_map(pa_type):
        key_type = _arrow_type_to_glue(pa_type.key_type, warnings)
        item_type = _arrow_type_to_glue(pa_type.item_type, warnings)
        return f"map<{key_type},{item_type}>"
    if patypes.is_struct(pa_type):
        parts = []
        for idx, field in enumerate(pa_type):
            field_name = field.name or f"field{idx}"
            parts.append(f"{field_name}:{_arrow_type_to_glue(field.type, warnings)}")
        return f"struct<{','.join(parts)}>"

    warnings.append(f"Falling back to string for unsupported Arrow type '{pa_type}'.")
    return "string"


def _build_columns(
    schema: pa.Schema, partition_names: set[str], warnings: List[str]
) -> tuple[list[dict], dict[str, str]]:
    """Separate regular and partition columns while preserving their types.

    Parameters
    ----------
    schema
        Full dataset schema obtained from PyArrow.
    partition_names
        Set of column names that should be treated as partitions.
    warnings
        Collector for type-mapping fallbacks.

    Returns
    -------
    tuple[list[dict], dict[str, str]]
        The Glue column definitions and a mapping ``partition_name -> type`` for
        the requested partitions.
    """
    columns: list[dict] = []
    partition_types: dict[str, str] = {}
    seen: set[str] = set()

    for field in schema:
        name = field.name
        if not name or name in seen:
            continue
        seen.add(name)
        glue_type = _arrow_type_to_glue(field.type, warnings)
        if name in partition_names:
            partition_types[name] = glue_type
            continue
        columns.append({"Name": name, "Type": glue_type})

    return columns, partition_types


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point that materialises the Glue create-table JSON payload."""
    parser = argparse.ArgumentParser(
        description=(
            "Create Glue table input JSON for a Parquet dataset so it can be "
            "registered without manual schema encoding."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Local path to the Parquet dataset root")
    parser.add_argument("--output-json", required=True, help="Path where the JSON definition will be written")
    parser.add_argument("--table-name", required=True, help="Glue table name")
    parser.add_argument("--s3-location", required=True, help="S3 prefix where the dataset lives")
    parser.add_argument(
        "--partition-cols",
        default="",
        help="Comma-separated list of partition column names (Hive style)",
    )
    parser.add_argument(
        "--classification",
        default="parquet",
        help="Glue classification value (default: parquet)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # Inspect the dataset with Hive-style partitioning awareness so we can
    # differentiate between physical partitions and actual columns.
    dataset = ds.dataset(args.dataset_root, format="parquet", partitioning="hive")
    schema = dataset.schema

    partition_names = {name.strip() for name in args.partition_cols.split(",") if name.strip()}
    warnings: List[str] = []
    columns, partition_types = _build_columns(schema, partition_names, warnings)

    partition_keys: list[dict] = []
    for name in args.partition_cols.split(","):
        pname = name.strip()
        if not pname:
            continue
        ptype = partition_types.get(pname, "string")
        if pname not in partition_types:
            warnings.append(
                f"Partition column '{pname}' not found in dataset schema; defaulting Glue type to string."
            )
        partition_keys.append({"Name": pname, "Type": ptype})

    location = args.s3_location.rstrip("/") + "/"

    table_input = {
        "Name": args.table_name,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "classification": args.classification,
            "EXTERNAL": "TRUE",
        },
        "StorageDescriptor": {
            "Columns": columns,
            "Location": location,
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                "Parameters": {"serialization.format": "1"},
            },
        },
        "PartitionKeys": partition_keys,
    }

    if partition_keys:
        # Glue needs this flag to understand Hive-style folder layouts.
        table_input["StorageDescriptor"]["StoredAsSubDirectories"] = True

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table_input, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # Emit any lossy conversions or schema mismatches collected earlier.
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
