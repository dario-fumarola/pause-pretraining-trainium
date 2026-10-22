#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export USE_TORCH_XLA=0
python -m pytest -q
python -m compileall -q train scripts tests
