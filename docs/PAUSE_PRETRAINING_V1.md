# Protocol: pause pretraining from initialization

Experiment **pause-pretraining-qwen3-0.6b-v1**, October 22, 2026. The current assignment stops after implementation, CPU tests, bounded whole-instance qualification and a measured budget. No large scientific run is launched.

## Hypothesis

At fixed architecture, initial weights, ordered real tokens and optimizer updates, pauses available from initialization improve held-out sequential computation more than vanilla pretraining or pauses introduced later. The utility signal is loss on subsequent real tokens, with no external teacher, intermediate reasoning labels or RL.

| Condition | Updates 0–511 | Updates 512–1023 | Primary inference |
|---|---|---|---|
| Vanilla | Ordinary NTP | Ordinary NTP | No pauses |
| Early | Pauses | Pauses | Pauses |
| Late | Ordinary NTP | Pauses, retaining Adam state | Pauses |

Early receives twice as much pause exposure as late. A difference supports the complete early recipe; it cannot isolate a pure timing effect. A later matched-exposure control must vary when equal pause-exposure blocks occur. This short scratch pilot does not compare mature pretraining ages.

Relevant prior art: [Goyal et al., ICLR 2024](https://arxiv.org/abs/2310.02226). This is a mechanism/onset pilot, not a novelty claim about pause tokens or self-generated textual thoughts. The [completed 1.7B results](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/main/docs/OVERNIGHT_RESULTS_20261022.md) remain frozen.

## Model, mechanism and objective

Use the [official Qwen3-0.6B-Base architecture](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json), saved in `configs/qwen3_0.6b_architecture.json`: 596,049,920 tied parameters, 28 layers, hidden size 1024, MLP 3072, 16 query heads, 8 KV heads, head dimension 128. Initialize using the standard HF initializer and a paired seed. Load no pretrained weights. The fixed existing tokenizer is byte-identical to the official 0.6B tokenizer; its hash is recorded.

Use existing tokenizer-undefined vocabulary row 151935 for pause inputs in every condition. Do not resize embeddings or add parameters. Keep the full vocabulary in likelihood denominators. Suppress the reserved row only in greedy generation, identically for every condition.

Each window has 513 real tokens: one context token and 512 scored successors. Insert 16 pauses after real-token counts 128, 256, 384 and 512, before a real successor. The trunk processes 576 positions versus 512. Never truncate real context or add unused trailing pauses. Use monotonic expanded RoPE positions.

For real target j, let p(j) be its expanded position. Predict it from hidden position p(j) − 1, including the **last pause** at boundaries. Give pauses no targets. Normalize loss by exactly the 512 real targets per window and the global batch; real EOS remains scored. Gradients flow through pause activations. Compute the vocabulary head only at real prediction positions. These are extra causal positions, not sixteen serial full-model invocations.

BF16 parameters/gradients; FP32 Adam masters/moments, rotary buffers, logits and CE. AdamW: peak LR 3e-4, betas (0.9,0.95), epsilon 1e-8, weight decay 0.1, global gradient clip 1. Linear warmup 64 updates, cosine decay to 3e-5 at update 1023. One ZeRO optimizer owns gradient reduction. Retain all optimizer state at the late switch. No arm-specific tuning.

## Proposed pilot data and size — not launched

Two paired initialization/data-order seeds, 2201 and 2202. Use the whole existing instance: 64 data-parallel ranks on 16 chips, microbatch 1, accumulation 2, global batch 128. Run 1024 updates per fit. This is **67,108,864 scored real targets** in 131,072 windows per condition; 402,653,184 across six fits, not that many distinct source tokens. This replaces the initial single-core topology while preserving the proposed per-fit token exposure.

For each seed, all three conditions share identical initial parameters, source windows/order, batch boundaries, source loss weights, learning-rate schedule and stopping point. Schedule by optimizer updates/real-token exposure, never expanded positions.

Every four windows: three ordinary text windows and one deterministic task. Ordinary windows are equal parts FineMath and FineWeb-Edu. Freeze source revisions, unique document IDs, tokenizer/config hashes, windows and ordered manifests before fitting. Split by source document before windowing. The supplied preparer uses retained QUALIFICATION sources for hardware checks only; it does not release old DEV/CONFIRM for training.

Task: starting from x in {0,…,96}, apply an ordered list of x <- (a*x+b) mod 97, with a in {2,3,5,7}. Supply only the final answer. Train equal counts of depths 4, 8 and 12. Hold out new programs at those depths; depth 16 tests length transfer. Neutral left newlines align the first answer token at real index 128, computed from the prompt alone. No intermediate states are supplied. Exclude exact program-content duplicates across TRAIN/DEV/CONFIRM using hashes of initial state plus operations, not RNG namespaces alone.

Before fitting, seal 4096 programs per trained depth, 2048 at depth 16, and 2048 unique text documents per domain. A separate 512-item DEV panel supports numerical diagnostics only. Terminal checkpoints are fixed in advance; never select using held-out outcomes.

## Endpoints and falsification

Primary: exact integer-answer accuracy, macro-averaged across depths 4, 8 and 12, from prefix-only greedy generation with a fixed four-token cap. Decode through first EOS/newline. Invalid or missing answers fail. Also report answer NLL, accuracy by depth and ordinary text NLL per domain. Task-format token likelihood is not a computation result.

Pair comparisons on the same program/document; bootstrap whole programs/source documents, 5000 draws, seed 17, percentile 95% intervals. Two training seeds are replications. Item intervals are conditional on their trained models, not confidence intervals over training runs.

Mechanism gates, required in both seeds:

1. Early with pauses beats vanilla by at least 5 percentage points, with a positive lower paired 95% bound.
2. Early with pauses beats the same early model with pauses removed by at least 2 points and a positive lower bound. Removal also shifts positions/input distribution; this is pathway dependence, not complete mechanistic identification.
3. Early ordinary text NLL is no more than 0.02 nats/token worse than vanilla in each domain. Report actual values and uncertainty.

Timing contrast: early minus late must have a positive lower bound in both seeds to support the complete early recipe. Report failure independently of the mechanism gates. Report every condition with pauses present and absent, including vanilla's untrained pause path. Report depth 16 regardless of direction.

Equal parameters and real tokens do **not** equalize FLOPs. Save one additional early checkpoint at floor(1024 × vanilla-step-FLOPs / early-step-FLOPs), fixed before scientific outcomes. The analytical matmul estimate gives a 1.1079079 ratio and **924 completed early updates**. Compare it to terminal vanilla: early has fewer real targets and no more estimated training FLOPs. This secondary comparison changes training age/LR. All main contrasts use terminal update 1024.

Measure teacher-forced scoring throughput, prefix-only cached generation latency, expanded KV positions and estimated inference FLOPs. Specify training-plus-inference budgets at a fixed request count before claiming efficiency. A same-token accuracy win alone does not establish a FLOP advantage, autonomous textual thought or general reasoning.

## Qualification and resource boundary

Focused CPU tests check target alignment, no dropped tokens, ordinary HF CE agreement, causality, gradients through actual pause inputs, optimizer continuity/recovery, cached generation across pause boundaries and exact distributed source order.

Native qualification uses all **16 existing chips / 64 logical cores**, full 0.6B models, six updates per condition, global batch 128, microbatch 1, accumulation 2, 768 shared source windows, and a late switch at update 3. Compare each rank's initial CE against its FP32 CPU reference from identical BF16-quantized weights; tolerance 0.03 nats/token. Require finite losses/gradient norms, actual LR, changed FP32 masters, one reducer, and equal full model replicas. Save the complete model and all 64 optimizer shards immediately before the switch; replay the last three steps and require bitwise equality of every model, master, moment and counter on every rank.

The owner explicitly requests the whole instance. The revised common qualification ceiling is **32 assigned chip-hours**, including all earlier stopped attempts, at most six total attempts and at most 45 minutes per whole-instance attempt. Four concurrent compiler processes bound host memory. Preserve 35 GiB free and 512 GiB host available. Do not delete old assets, buy capacity or change the accelerator installation.

Only new temporary qualification assets go under `/run/user/1000/pause-pretraining-v1`. Publish numerical receipts, never source token IDs or native checkpoints. The first three single-core attempts are closed and cost 0.4046664154 assigned chip-hours; all remain in the ledger.

A large run remains disabled. Finish actual qualification, throughput and the budget first. Scientific source/split manifests and exact evaluation/export/storage paths must be ready before a later launch. Learned pause timing, self-generated states, matched exposure and internalization are follow-up experiments.
