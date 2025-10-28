#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Wrapper script: apply_buoy_wind_height_correction.sh
# -----------------------------------------------------------------------------
# Runs the buoy wind height correction helper inside a dockerised Python
# environment to guarantee consistent dependencies across hosts.
# -----------------------------------------------------------------------------

set -euo pipefail

IMAGE="python:3.11-slim"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

AWS_PROFILE_NAME=${AWS_PROFILE:-default}

declare -a DOCKER_ARGS=(
  --rm
  -v "${REPO_ROOT}":/work
  -w /work
  -e "AWS_PROFILE=${AWS_PROFILE_NAME}"
)

if [ -d "${HOME}/.aws" ]; then
  DOCKER_ARGS+=(-v "${HOME}/.aws:/root/.aws")
fi

PY_ARGS=""
if [ $# -gt 0 ]; then
  PY_ARGS="$(printf ' %q' "$@")"
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash -lc \
  "pip install --no-cache-dir 'pyarrow==16.1.0' 'awscli==1.34.11' >/tmp/pip.log && python scripts/aggregation/apply_buoy_wind_height_correction.py${PY_ARGS}"
