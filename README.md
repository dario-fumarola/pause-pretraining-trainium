# Pause pretraining on Trainium

Can a language model learn to use extra computation more effectively when it is available from the start of pretraining?

This focused experiment uses the Qwen3-0.6B architecture **from random initialization**, with three conditions: ordinary next-token pretraining, fixed pauses from initialization, and the same pauses introduced halfway through training. Insert 16 unlabeled pause positions every 128 real tokens. No teacher model or RL.

Real-token exposure, data order and parameter count match. Training and inference compute are measured separately. A deterministic modular-composition task tests computation; ordinary text prediction tests the cost. Early versus late also changes pause exposure, so this pilot cannot isolate a pure timing effect.

The current assignment is protocol, implementation, CPU tests, a bounded **64-rank qualification on all 16 existing Trainium chips**, measured throughput and a justified budget. **No large scientific run has launched.**

[Protocol](docs/PAUSE_PRETRAINING_V1.md) · [Status](STATUS.md) · [Commands](docs/QUICKSTART.md)

The completed 1.7B evidence remains in [useful-thoughts-trainium](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/main/docs/OVERNIGHT_RESULTS_20261022.md). Its original results and native checkpoints are preserved. This repository contains only the new experiment and the small runtime primitives it reuses.

## CPU checks

```bash
source scripts/env.sh
bash scripts/run_checks.sh
```

The environment script reuses the existing node environment. It does not install or upgrade Neuron. Required Python packages: PyTorch, Transformers (qualified version 5.15.1), NumPy and pytest; native execution additionally requires the installed PyTorch/XLA and Neuron stack. CPU tests do not qualify hardware.

No training source text/token IDs, model weights, optimizer states, compiler caches or credentials are published here. Numerical receipts include hashes and measurements.
