#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

"""Materialise an Athena view that vertically stacks two sources with schema harmonisation.

The helper issues the required AWS Glue and Athena commands so downstream pipelines can
treat two compatible tables or views as a single unioned dataset. Column metadata is
queried from ``information_schema`` to build a consistent projection, optional renames
are applied to close schema gaps, and the resulting ``SELECT`` statement is either
previewed or executed directly against Athena.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Tuple


def log(message: str) -> None:
    """Emit a timestamped log line to stdout."""

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}", flush=True)


def run_aws(profile: str, region: str, args: List[str]) -> str:
    """Execute an AWS CLI command and return stdout, raising richly on failure.

    The helper injects optional profile and region selectors, ensuring every command
    honours the user-supplied execution context. When the CLI exits with a non-zero
    status the exception surfaces the full command line and the stderr payload, which
    greatly accelerates debugging in automated pipelines.
    """

    cmd: List[str] = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    if region:
        cmd += ["--region", region]
    cmd += args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "AWS CLI failed (exit {}): {}\n{}".format(
                proc.returncode,
                " ".join(shlex.quote(part) for part in cmd),
                proc.stderr.strip(),
            )
        )
    return proc.stdout


def wait_for_query(profile: str, region: str, query_id: str) -> None:
    """Poll Athena until the given query finishes, propagating failures immediately."""

    while True:
        raw = run_aws(profile, region, ["athena", "get-query-execution", "--query-execution-id", query_id])
        data = json.loads(raw)
        status = data["QueryExecution"]["Status"]
        state = status["State"]
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED"}:
            reason = status.get("StateChangeReason", "(no reason provided)")
            raise RuntimeError(f"Athena query {query_id} ended with {state}: {reason}")
        time.sleep(3)


def fetch_columns(profile: str, region: str, results_s3: str, database: str, table: str) -> List[str]:
    """Return the ordered column list for a Glue table or view leveraging information_schema."""

    sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{database.lower()}' "
        f"AND table_name = '{table.lower()}' ORDER BY ordinal_position"
    )
    raw = run_aws(
        profile,
        region,
        [
            "athena",
            "start-query-execution",
            "--query-execution-context",
            f"Database={database}",
            "--result-configuration",
            f"OutputLocation={results_s3}",
            "--query-string",
            sql,
        ],
    )
    query_id = json.loads(raw)["QueryExecutionId"]
    wait_for_query(profile, region, query_id)
    results_raw = run_aws(profile, region, ["athena", "get-query-results", "--query-execution-id", query_id])
    results = json.loads(results_raw)
    rows = results.get("ResultSet", {}).get("Rows", [])[1:]
    columns: List[str] = []
    for row in rows:
        data = row.get("Data", [])
        if not data:
            continue
        value = data[0].get("VarCharValue")
        if value:
            columns.append(value)
    return columns


def parse_rename_spec(spec: str, label: str) -> Dict[str, str]:
    """Parse comma-separated rename directives into a mapping, validating identifiers."""

    mapping: Dict[str, str] = {}
    if not spec:
        return mapping
    token_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for raw_pair in spec.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid rename mapping '{pair}' for {label}; expected old=new syntax")
        src, dest = (part.strip() for part in pair.split("=", 1))
        if not src or not dest:
            raise ValueError(f"Empty source or destination in rename mapping '{pair}' for {label}")
        if not token_re.match(dest):
            raise ValueError(f"Invalid identifier '{dest}' in rename mapping for {label}")
        if src in mapping:
            raise ValueError(f"Duplicate rename mapping for column '{src}' in {label}")
        mapping[src] = dest
    return mapping


def build_final_mapping(columns: List[str], rename_map: Dict[str, str], label: str) -> Dict[str, str]:
    """Compute the post-rename mapping that keeps track of the original column names."""

    final: Dict[str, str] = {}
    for col in columns:
        final_name = rename_map.get(col, col)
        if final_name in final and final[final_name] != col:
            raise ValueError(
                f"Multiple columns in {label} map to '{final_name}' (conflicts between '{final[final_name]}' and '{col}')"
            )
        if final_name not in final:
            final[final_name] = col
    return final


def build_select_clause(mapping: Dict[str, str], final_columns: List[str]) -> str:
    """Generate the SELECT clause that re-aliases columns back to the harmonised names."""

    parts: List[str] = []
    for col in final_columns:
        source_col = mapping[col]
        if source_col == col:
            parts.append(f"  {source_col}")
        else:
            parts.append(f"  {source_col} AS {col}")
    return ",\n".join(parts)


def ensure_database(profile: str, region: str, database: str) -> None:
    """Guarantee the Glue database exists before issuing DDL statements."""

    try:
        run_aws(profile, region, ["glue", "get-database", "--name", database])
    except RuntimeError:
        log(f"Creating Glue database: {database}")
        run_aws(
            profile,
            region,
            [
                "glue",
                "create-database",
                "--database-input",
                json.dumps({"Name": database}),
            ],
        )


def create_view(
    profile: str,
    region: str,
    results_s3: str,
    view_db: str,
    view_table: str,
    select_sql: str,
) -> None:
    """Issue the drop/create sequence materialising the harmonised view in Athena."""

    ensure_database(profile, region, view_db)
    drop_view_sql = f"DROP VIEW IF EXISTS {view_db}.{view_table}"
    raw = run_aws(
        profile,
        region,
        [
            "athena",
            "start-query-execution",
            "--query-string",
            drop_view_sql,
            "--query-execution-context",
            f"Database={view_db}",
            "--result-configuration",
            f"OutputLocation={results_s3}",
        ],
    )
    drop_view_id = json.loads(raw)["QueryExecutionId"]
    wait_for_query(profile, region, drop_view_id)

    # Drop a potential table counterpart as legacy runs may have left one behind.
    drop_table_sql = f"DROP TABLE IF EXISTS {view_db}.{view_table}"
    raw = run_aws(
        profile,
        region,
        [
            "athena",
            "start-query-execution",
            "--query-string",
            drop_table_sql,
            "--query-execution-context",
            f"Database={view_db}",
            "--result-configuration",
            f"OutputLocation={results_s3}",
        ],
    )
    drop_table_id = json.loads(raw)["QueryExecutionId"]
    wait_for_query(profile, region, drop_table_id)

    formatted = (
        f"CREATE OR REPLACE VIEW {view_db}.{view_table} AS\n{select_sql}\n;"
    )
    raw = run_aws(
        profile,
        region,
        [
            "athena",
            "start-query-execution",
            "--query-string",
            formatted,
            "--query-execution-context",
            f"Database={view_db}",
            "--result-configuration",
            f"OutputLocation={results_s3}",
        ],
    )
    query_id = json.loads(raw)["QueryExecutionId"]
    wait_for_query(profile, region, query_id)


def main(argv: List[str]) -> int:
    """Parse CLI arguments, orchestrate column discovery, and create the view."""

    parser = argparse.ArgumentParser(description="Concatenate two Athena tables into a harmonised view")
    parser.add_argument("--source-a", required=True, help="First source table/view (database.table)")
    parser.add_argument("--source-b", required=True, help="Second source table/view (database.table)")
    parser.add_argument("--view", required=True, help="Target view (database.table)")
    parser.add_argument("--results-s3", required=True, help="S3 path for Athena query results")
    parser.add_argument("--rename-a", default="", help="Comma-separated rename mappings for source A")
    parser.add_argument("--rename-b", default="", help="Comma-separated rename mappings for source B")
    parser.add_argument(
        "--union-type",
        default="all",
        choices=["all", "distinct"],
        help="Use UNION ALL (default) or UNION (distinct) when stacking rows",
    )
    parser.add_argument("--profile", default="", help="AWS CLI profile")
    parser.add_argument("--region", default="", help="AWS region")
    parser.add_argument("--preview", action="store_true", help="Print the generated SQL and exit")

    args = parser.parse_args(argv)

    def split_fqn(value: str, label: str) -> Tuple[str, str]:
        """Split fully qualified names in the expected database.table format."""

        if "." not in value:
            raise ValueError(f"Expected {label} in database.table format, got '{value}'")
        db, table = value.split(".", 1)
        if not db or not table:
            raise ValueError(f"Invalid {label} '{value}'")
        return db, table

    source_a_db, source_a_tbl = split_fqn(args.source_a, "--source-a")
    source_b_db, source_b_tbl = split_fqn(args.source_b, "--source-b")
    view_db, view_tbl = split_fqn(args.view, "--view")

    rename_a = parse_rename_spec(args.rename_a, "source A")
    rename_b = parse_rename_spec(args.rename_b, "source B")

    log(f"Fetching column metadata for source A: {args.source_a}")
    columns_a = fetch_columns(args.profile, args.region, args.results_s3, source_a_db, source_a_tbl)
    if not columns_a:
        raise ValueError(f"No columns found for {args.source_a}")

    log(f"Fetching column metadata for source B: {args.source_b}")
    columns_b = fetch_columns(args.profile, args.region, args.results_s3, source_b_db, source_b_tbl)
    if not columns_b:
        raise ValueError(f"No columns found for {args.source_b}")

    mapping_a = build_final_mapping(columns_a, rename_a, "source A")
    mapping_b = build_final_mapping(columns_b, rename_b, "source B")

    # Retain only the columns that survive the rename step on both sources.
    final_columns = [col for col in mapping_a.keys() if col in mapping_b]
    if not final_columns:
        raise ValueError("The two sources have no overlapping columns after applying rename mappings")

    # Surface the asymmetric columns so operators can review what was discarded.
    dropped_a = [col for col in mapping_a.keys() if col not in mapping_b]
    dropped_b = [col for col in mapping_b.keys() if col not in mapping_a]

    if dropped_a:
        log("Columns dropped from source A: " + ", ".join(dropped_a))
    if dropped_b:
        log("Columns dropped from source B: " + ", ".join(dropped_b))

    select_a = build_select_clause(mapping_a, final_columns)
    select_b = build_select_clause(mapping_b, final_columns)

    # ``UNION`` switches between deduplicating rows and leaving them untouched.
    union_keyword = "UNION ALL" if args.union_type == "all" else "UNION"

    # Compose the SELECT/UNION statement that backs the view definition.
    select_sql = (
        "SELECT\n"
        f"{select_a}\n"
        f"FROM {source_a_db}.{source_a_tbl}\n"
        f"{union_keyword}\n"
        "SELECT\n"
        f"{select_b}\n"
        f"FROM {source_b_db}.{source_b_tbl}"
    )

    if args.preview:
        # Developers often inspect the generated SQL before executing the DDL.
        print(
            f"CREATE OR REPLACE VIEW {view_db}.{view_tbl} AS\n{select_sql}\n;"
        )
        return 0

    log(f"Creating or replacing view {args.view}")
    create_view(args.profile, args.region, args.results_s3, view_db, view_tbl, select_sql)
    log(f"View {args.view} created successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
