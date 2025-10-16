# Filtered Athena View Builder

## Purpose
`scripts/aggregation/create_filtered_view.sh` creates or replaces an Athena view that filters or reshapes the output of the pivot-and-join workflow using user-supplied SQL. The script accepts a SQL template or clause, injects the fully qualified source table name, and submits the resulting `CREATE OR REPLACE VIEW` statement through the AWS CLI. It is typically used to derive quality-controlled subsets (for example, filtered pivot views) before partitioning or downstream analytics.

## Prerequisites
- AWS CLI v2 with permissions to run Athena queries and manage Glue views.
- A source table or view produced by earlier aggregation steps (`join_pivoted_tables.sh` plus optional enrichments).
- An Athena-compatible SQL snippet saved on disk. Tokens or predicates inside the snippet should reference the column names exposed by the source table.

## Key Options
- `--source <db.table>`: Fully-qualified source table/view to filter.
- `--view <db.view>`: Target view name that will be created or replaced.
- `--sql-file <path>`: SQL fragment or template defining the filter logic. Supports the `{{SOURCE_TABLE}}` placeholder to embed the source table name inside bespoke SELECT statements.
- `--results-s3 <s3://...>`: Athena scratch output bucket; required unless `--preview` is used.
- `--preview`: Print the generated statement instead of executing it.
- `--profile` / `--region`: AWS CLI profile and region.

## SQL File Conventions
- If the file contains `{{SOURCE_TABLE}}`, the placeholder is replaced with the fully qualified table name and the script executes the file contents verbatim.
- When the first non-blank token is `SELECT` or `WITH`, the snippet is run as provided.
- Snippets beginning with `WHERE`, `AND`, or `OR` are appended to `SELECT * FROM <source>` to build the view.
- Bare boolean expressions are wrapped in a `WHERE` clause automatically.
- An empty file materialises the view as a straight copy of the source table.

## Typical Usage
```bash
scripts/aggregation/create_filtered_view.sh \
  --source <db.source_table> \
  --view <db.filtered_view> \
  --sql-file sql/<filter_clause.sql> \
  --results-s3 s3://<bucket>/<athena_results_prefix>/ \
  --profile <aws_profile> \
  --region <aws_region>
```
This command creates a filtered view suitable for any downstream partition or aggregation step; adjust the placeholders to reflect your own schema and storage layout.

## Behaviour Notes
- Glue databases referenced in `--view` are created automatically if they do not already exist.
- The script trims trailing semicolons and Windows carriage returns, ensuring the query executes cleanly.
- `--preview` is useful for verifying templated SQL (for example, when the snippet expands to a long CTE) without running Athena jobs.

## Troubleshooting
- *`--sql-file` not found*: Confirm that the path is relative to the repository root or provide an absolute path.
- *Placeholder missing*: If the SQL needs the source table name within joins or subqueries, always reference `{{SOURCE_TABLE}}`; otherwise the snippet must be self-contained.
- *Athena permission errors*: Verify the profile has rights to read the SQL file's target bucket and to write to the results bucket.

## File Reference
- `scripts/aggregation/create_filtered_view.sh`
