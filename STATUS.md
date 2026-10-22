# Current status

October 22, 2026. New focused repository; the predecessor 1.7B evidence is preserved.

Protocol and implementation are prepared. The predecessor full suite passes 2,086 CPU tests; this repository runs its own focused numerical/causal tests. The owner now requests all 16 chips / 64 logical cores for the small native qualification. The previous three single-core attempts are closed and retained, totaling 0.4046664154 assigned chip-hours. Two were stopped for engineering fixes; the third was stopped to apply the whole-instance instruction. No large run is launched.

Qualification output root: `/run/user/1000/pause-pretraining-v1`. The full-instance configuration remains bounded to six updates per condition, global batch 128, microbatch 1, accumulation 2, and exact full-state replay at the late switch. The revised common qualification cap is 32 assigned chip-hours including the single-core attempts. Preserve 35 GiB free and 512 GiB host available.

Next: run this repository's CPU checks and CPU reference, then the bounded whole-instance supervisor. Measure actual distributed throughput and inference latency, publish numerical receipts and a budget, and close all workers. Scientific data preparation and a large training run remain disabled.
