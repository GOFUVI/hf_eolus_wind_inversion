#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
# Created: 2025-10-16
# Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
# -----------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: prepare_finetune_config.sh --base PATH --output PATH --target-speed-col NAME --target-dir-col NAME [--force]

Copies the base model configuration if needed and updates the target column
names inside the model definition.
USAGE
  exit 1
}

BASE_CONFIG=""
OUTPUT_CONFIG=""
TARGET_SPEED=""
TARGET_DIR=""
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_CONFIG="$2"; shift 2 ;;
    --base=*)
      BASE_CONFIG="${1#*=}"; shift ;;
    --output)
      OUTPUT_CONFIG="$2"; shift 2 ;;
    --output=*)
      OUTPUT_CONFIG="${1#*=}"; shift ;;
    --target-speed-col)
      TARGET_SPEED="$2"; shift 2 ;;
    --target-speed-col=*)
      TARGET_SPEED="${1#*=}"; shift ;;
    --target-dir-col)
      TARGET_DIR="$2"; shift 2 ;;
    --target-dir-col=*)
      TARGET_DIR="${1#*=}"; shift ;;
    --force)
      FORCE=true; shift ;;
    --help|-h)
      usage ;;
    --)
      shift; break ;;
    -* )
      echo "Error: unknown option '$1'" >&2
      usage ;;
    * )
      echo "Error: unexpected positional argument '$1'" >&2
      usage ;;
  esac
done

if [[ -z "$BASE_CONFIG" || -z "$OUTPUT_CONFIG" || -z "$TARGET_SPEED" || -z "$TARGET_DIR" ]]; then
  echo "Error: --base, --output, --target-speed-col and --target-dir-col are required." >&2
  usage
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Error: base config '$BASE_CONFIG' not found." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CONFIG")"

if [[ ! -f "$OUTPUT_CONFIG" || "$FORCE" == true ]]; then
  cp "$BASE_CONFIG" "$OUTPUT_CONFIG"
  echo ">>> Copied $BASE_CONFIG to $OUTPUT_CONFIG <<<"
else
  echo ">>> Reusing $OUTPUT_CONFIG <<<"
fi

python3 - <<PY
import json
from pathlib import Path
cfg_path = Path("$OUTPUT_CONFIG")
cfg = json.loads(cfg_path.read_text())
# Drop any stale root-level settings to avoid duplicating configuration knobs
cfg.pop("target_speed_col", None)
cfg.pop("target_dir_col", None)
model = cfg.get("model")
if not isinstance(model, dict):
    model = {}
model["target_speed_col"] = "$TARGET_SPEED"
model["target_dir_col"] = "$TARGET_DIR"
cfg["model"] = model
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
PY

echo ">>> Updated $OUTPUT_CONFIG with target columns $TARGET_SPEED / $TARGET_DIR inside model <<<"
