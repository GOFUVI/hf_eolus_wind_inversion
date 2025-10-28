#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-21
# Disclaimer: This script is provided as-is for research workflows and must be
# validated before any operational deployment.
# -----------------------------------------------------------------------------

"""Materialise a buoy-linked pivot table with wind speeds converted between heights."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def log(message: str, *, logfile: Path | None = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} - {message}"
    print(line, flush=True)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_aws(
    args: Sequence[str],
    *,
    profile: str,
    region: str,
    logfile: Path | None = None,
) -> str:
    cmd: List[str] = ["aws"]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    cmd.extend(args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(f"Running: {' '.join(shlex.quote(part) for part in cmd)}\n")
            if proc.stdout:
                handle.write(proc.stdout + "\n")
            if proc.stderr:
                handle.write(proc.stderr + "\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"AWS CLI failed (exit {proc.returncode}): "
            f"{' '.join(shlex.quote(part) for part in cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def wait_for_query(
    profile: str,
    region: str,
    query_id: str,
    *,
    logfile: Path | None = None,
) -> None:
    while True:
        raw = run_aws(
            ["athena", "get-query-execution", "--query-execution-id", query_id],
            profile=profile,
            region=region,
            logfile=logfile,
        )
        data = json.loads(raw)
        state = data["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED"}:
            reason = data["QueryExecution"]["Status"].get("StateChangeReason", "(no reason provided)")
            raise RuntimeError(f"Athena query {query_id} ended with {state}: {reason}")
        time.sleep(3)


def fetch_columns(
    source_db: str,
    source_table: str,
    *,
    profile: str,
    region: str,
    results_s3: str,
    logfile: Path | None = None,
) -> List[Tuple[str, str]]:
    sql = (
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = '{source_db.lower()}' "
        f"AND table_name = '{source_table.lower()}' ORDER BY ordinal_position"
    )
    start_raw = run_aws(
        [
            "athena",
            "start-query-execution",
            "--query-execution-context",
            f"Database={source_db}",
            "--result-configuration",
            f"OutputLocation={results_s3}",
            "--query-string",
            sql,
        ],
        profile=profile,
        region=region,
        logfile=logfile,
    )
    query_id = json.loads(start_raw)["QueryExecutionId"]
    wait_for_query(profile, region, query_id, logfile=logfile)
    results_raw = run_aws(
        ["athena", "get-query-results", "--query-execution-id", query_id],
        profile=profile,
        region=region,
        logfile=logfile,
    )
    rows = json.loads(results_raw).get("ResultSet", {}).get("Rows", [])[1:]
    columns: List[Tuple[str, str]] = []
    for row in rows:
        data = row.get("Data", [])
        if len(data) >= 2:
            name = data[0].get("VarCharValue")
            dtype = data[1].get("VarCharValue")
            if name:
                columns.append((name, dtype))
    return columns


def parse_db_table(spec: str, label: str) -> Tuple[str, str]:
    if "." not in spec:
        raise ValueError(f"{label} must be provided as database.table, got '{spec}'")
    db, table = spec.split(".", 1)
    if not db or not table:
        raise ValueError(f"{label} contains empty database or table in '{spec}'")
    return db, table


def build_select_lines(
    columns: Iterable[Tuple[str, str]],
    *,
    correction_column: str,
    factor: float,
) -> List[str]:
    lines: List[str] = []
    for name, _dtype in columns:
        if name == correction_column:
            lines.append(
                f"  CASE WHEN {name} IS NULL OR {name} <= -9000 THEN {name} "
                f"ELSE {name} * {factor:.15f} END AS {name}"
            )
        else:
            lines.append(f"  {name}")
    return lines


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Materialise a buoy pivot dataset with wind speeds converted to a target height.",
    )
    parser.add_argument("--source", required=True, help="Source table (database.table)")
    parser.add_argument("--target-table", required=True, help="Target table (database.table)")
    parser.add_argument("--s3-output", required=True, help="S3 prefix for the CTAS output")
    parser.add_argument("--results-s3", help="S3 prefix for Athena spill results")
    parser.add_argument("--source-height", type=float, required=True, help="Measurement height (m)")
    parser.add_argument("--target-height", type=float, required=True, help="Target height (m)")
    parser.add_argument(
        "--roughness-length",
        type=float,
        required=True,
        help="Surface roughness length (m) for the logarithmic wind profile",
    )
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "default"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", ""))
    parser.add_argument("--log-dir", default=".")

    args = parser.parse_args(argv)

    source_db, source_table = parse_db_table(args.source, "--source")
    target_db, target_table = parse_db_table(args.target_table, "--target-table")
    s3_output = args.s3_output.rstrip("/") + "/"
    if args.results_s3:
        results_s3 = args.results_s3.rstrip("/") + "/"
    else:
        raise ValueError("--results-s3 must be provided and point to an Athena spill prefix outside the output dataset.")
    if results_s3.startswith(s3_output):
        raise ValueError(
            f"--results-s3 ({results_s3}) must not reside under the output prefix ({s3_output}); "
            "Athena metadata files placed alongside the dataset corrupt downstream reads."
        )
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"apply_buoy_wind_height_correction_{target_table}.log"

    region = args.region
    if not region:
        proc = subprocess.run(
            ["aws", "--profile", args.profile, "configure", "get", "region"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            region = proc.stdout.strip()
    if not region:
        raise RuntimeError("AWS region not provided and not configured for the selected profile.")

    if args.source_height <= 0 or args.target_height <= 0 or args.roughness_length <= 0:
        raise ValueError("Heights and roughness length must be strictly positive.")

    factor = math.log(args.target_height / args.roughness_length) / math.log(
        args.source_height / args.roughness_length
    )

    log(
        f"Applying logarithmic wind correction from {args.source_height} m to {args.target_height} m "
        f"(z0={args.roughness_length} m, factor={factor:.6f})",
        logfile=log_file,
    )

    columns = fetch_columns(
        source_db,
        source_table,
        profile=args.profile,
        region=region,
        results_s3=results_s3,
        logfile=log_file,
    )
    if not columns:
        raise RuntimeError(f"Source table {source_db}.{source_table} has no columns.")

    select_lines = build_select_lines(columns, correction_column="wind_speed", factor=factor)
    select_body = ",\n".join(select_lines)

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".sql") as tmp:
        tmp.write(
            f"CREATE TABLE {target_db}.{target_table}\n"
            "WITH (\n"
            "  format='PARQUET',\n"
            f"  external_location='{s3_output}'\n"
            ")\n"
            "AS\n"
            "SELECT\n"
            f"{select_body}\n"
            f"FROM {source_db}.{source_table}\n"
        )
        sql_path = tmp.name

    try:
        log(
            f"Dropping existing Athena table {target_db}.{target_table} if present",
            logfile=log_file,
        )
        drop_raw = run_aws(
            [
                "athena",
                "start-query-execution",
                "--query-execution-context",
                f"Database={target_db}",
                "--result-configuration",
                f"OutputLocation={results_s3}",
                "--query-string",
                f"DROP TABLE IF EXISTS {target_db}.{target_table}",
            ],
            profile=args.profile,
            region=region,
            logfile=log_file,
        )
        drop_id = json.loads(drop_raw).get("QueryExecutionId")
        if drop_id:
            wait_for_query(args.profile, region, drop_id, logfile=log_file)

        log(f"Clearing S3 prefix {s3_output}", logfile=log_file)
        run_aws(
            ["s3", "rm", "--recursive", s3_output],
            profile=args.profile,
            region=region,
            logfile=log_file,
        )

        log(
            f"Submitting CTAS to materialise {target_db}.{target_table}",
            logfile=log_file,
        )
        create_raw = run_aws(
            [
                "athena",
                "start-query-execution",
                "--query-execution-context",
                f"Database={target_db}",
                "--result-configuration",
                f"OutputLocation={results_s3}",
                "--query-string",
                f"file://{sql_path}",
            ],
            profile=args.profile,
            region=region,
            logfile=log_file,
        )
        create_id = json.loads(create_raw)["QueryExecutionId"]
        wait_for_query(args.profile, region, create_id, logfile=log_file)

        log("Height-corrected buoy dataset materialised successfully", logfile=log_file)
    finally:
        try:
            os.unlink(sql_path)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
