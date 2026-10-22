#!/usr/bin/env bash
# Existing node only; do not modify the accelerator installation.
PAUSE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ubuntu/ema-trainium/.venv/bin:/opt/aws/neuron/bin:$PATH"
export PYTHONPATH="$PAUSE_REPO_ROOT:/home/ubuntu/dario/local/test-deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
