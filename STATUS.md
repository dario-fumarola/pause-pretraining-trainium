# Current status

October 22, 2026. Public repository: https://github.com/dario-fumarola/pause-pretraining-trainium. The separate 1.7B results remain preserved in useful-thoughts-trainium.

**13 focused CPU tests and compilation pass.** They cover causal real-token targets, HF agreement, pause gradients, optimizer recovery, cached generation and complete 64-rank data ordering. The predecessor's 2,086-test suite also passed before extraction; that is separate evidence.

**Whole-instance native qualification is running:** 64 ranks, all 16 chips observed, global batch 128, microbatch 1, accumulation 2, six updates per condition. Native parent PID **3274161**. Output: `/run/user/1000/pause-pretraining-v1/attempt-whole-001`. Read `run.log` and `resources.jsonl`; do not overlap another attempt. The common cap is 32 assigned chip-hours including all historical attempts, with a 45-minute per-attempt wall limit. Preserve 35 GiB free and 512 GiB available host memory. Four compiler slots bound CPU compiler memory.

The three previous single-core attempts are closed and retained, costing **0.4046664154 assigned chip-hours**. Two stopped for engineering fixes; the third stopped to apply the owner's whole-instance instruction. None establishes native qualification.

Pending: actual six-step model updates in all conditions, initial CPU/native parity, equal replicas, full 64-shard recovery across the late switch, training/scoring/cached-generation throughput, complete closure and a measured experiment budget. Compilation and process occupancy alone are not a pass.

**No large scientific run is launched.** Proposed later science preserves 67,108,864 real targets per fit using 1024 updates × global batch 128 × 512 targets; 402,653,184 across six fits. Scientific source/split manifests and the final budget remain prelaunch work. See the protocol and QUICKSTART.
