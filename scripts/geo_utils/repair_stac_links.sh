#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

# repair_stac_links.sh
# -----------------------------------------------------------------------------
# Ensure every STAC collection under a catalog directory links back to the
# shared catalog.json, mirroring the post-processing performed during catalog
# generation. The script runs the Python helper inside a slim Docker image so
# it can be invoked from environments lacking a local Python toolchain.
# -----------------------------------------------------------------------------

# usage prints the command synopsis and exits with the provided status.
usage() {
  cat <<'EOF'
Usage: repair_stac_links.sh [--catalog-dir DIR]

Options:
  --catalog-dir DIR   Catalog root containing catalog.json and sub-collections
                      (default: catalogs)
  -h, --help          Show this help message
EOF
}

catalog_dir="catalogs"

# Parse CLI arguments, keeping positional usage intentionally simple so the
# script stays copy/paste friendly when invoked from run_* pipelines.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --catalog-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --catalog-dir requires a value" >&2
        usage
        exit 1
      fi
      catalog_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

# Compute an absolute catalog path while enforcing that the provided directory
# lives under the project root; this allows Docker volume mounting to mirror
# the host layout inside /app for relative link reconstruction.
if [[ "$catalog_dir" = /* ]]; then
  candidate="$catalog_dir"
else
  candidate="${project_root}/${catalog_dir}"
fi

if [[ ! -d "$candidate" ]]; then
  echo "Error: catalog directory not found: $candidate" >&2
  exit 1
fi

abs_catalog_dir="$(cd "$candidate" && pwd)"

case "$abs_catalog_dir" in
  "$project_root"*)
    suffix="${abs_catalog_dir#$project_root}"
    container_catalog_dir="/app${suffix}"
    ;;
  *)
    echo "Error: catalog directory must reside within the project root (${project_root})" >&2
    exit 1
    ;;
esac

# Execute the Python helper inside a minimal interpreter container to avoid
# polluting the host environment with ad-hoc dependencies while guaranteeing a
# consistent runtime across machines.
docker run --rm \
  -v "${project_root}":/app \
  -w /app \
  -e PYTHONDONTWRITEBYTECODE=1 \
  python:3.11-slim \
  python scripts/geo_utils/repair_stac_links.py "$container_catalog_dir"
