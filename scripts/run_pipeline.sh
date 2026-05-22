#!/usr/bin/env bash
# Run Phase 1 pipeline from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT}:${ROOT}/sdk:${ROOT}/schemas:${ROOT}/intelligence/src"
export PATH="/opt/homebrew/bin:${PATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-atlas}"
mkdir -p "$MPLCONFIGDIR"

exec python scripts/run_pipeline.py "$@"
