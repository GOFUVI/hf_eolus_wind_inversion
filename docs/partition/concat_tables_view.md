# Concatenate Tables View Builder

## Purpose
`scripts/partition/concat_tables_view.sh` materialises an Athena view that vertically stacks two source tables or views after normalising their column names. The script aligns field names via optional rename mappings, discards non-overlapping columns, and issues a `CREATE OR REPLACE VIEW` so downstream jobs (e.g., partitioning) can treat the merged dataset as a single harmonised table.

## Prerequisites
- Python 3 with the AWS CLI v2 available on the system `PATH`.
- AWS credentials that permit Athena query execution and Glue catalog updates.
- Two Athena tables or views whose schemas overlap after optional renaming. The sources may themselves be views.
- An S3 bucket/prefix that Athena can use for query results (`--results-s3`).

## Key Options
- `--source-a <db.table>` / `--source-b <db.table>`: Fully-qualified names of the tables or views to union.
- `--rename-a <old=new,...>` / `--rename-b <old=new,...>`: Comma-separated rename mappings applied before matching columns. Identifiers must be valid Athena column names.
- `--view <db.table>`: Target view (database.table) that will be created or replaced.
- `--results-s3 <s3://...>`: Athena scratch output location used for metadata queries and the final DDL.
- `--union-type {all,distinct}`: Use `UNION ALL` (default) or `UNION` when combining the two SELECT statements.
- `--preview`: Print the generated SQL without executing it—helpful for inspection during pipeline development.
- `--profile` / `--region`: AWS CLI profile and region overrides.

## Behaviour Notes
- Column metadata is fetched from `information_schema.columns`; the script logs each fetch so failures pinpoint which source caused the issue.
- Only columns present in both sources after renaming are retained. Dropped columns are listed in the logs for transparency.
- Any Glue database referenced by `--view` is auto-created if missing.
- Errors from the AWS CLI propagate with context (exit code, command, and stderr) so misconfigurations are easy to diagnose.

## Typical Usage
```bash
scripts/partition/concat_tables_view.sh \
  --source-a <db.source_table_a> \
  --rename-a <rename_map_for_a> \
  --source-b <db.source_table_b> \
  --rename-b <rename_map_for_b> \
  --view <db.concatenated_view> \
  --results-s3 s3://<bucket>/<athena_results_prefix>/ \
  --profile <aws_profile> \
  --region <aws_region>
```
This command produces a harmonised view from any pair of compatible tables; customise the rename maps and identifiers to match your schema.

## Troubleshooting
- *`Invalid rename mapping`*: Ensure each mapping uses `old=new`, and that the new name is a valid identifier. Mappings that reference a column absent from the source are silently ignored because schema validation relies on Athena metadata; double-check spellings whenever a rename seems ineffective.
- *`No columns found`*: Confirm the database and table names exist and that the AWS profile has permissions to query `information_schema`.
- *Athena failures*: Review the error message (printed with the AWS CLI stderr) for missing permissions, nonexistent views, or bucket access issues.

## File Reference
- `scripts/partition/concat_tables_view.sh`
