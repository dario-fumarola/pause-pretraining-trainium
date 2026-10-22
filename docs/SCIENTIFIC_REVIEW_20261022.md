# Scientific review adopted — October 22

The owner's review correctly identifies undertraining as a major interpretive limit. The proposed first checkpoint has 67,108,864 real targets for 596,049,920 parameters, about 0.113 targets per parameter. It is a feasibility checkpoint, not an assumption that general language or arithmetic has been learned.

The current hardware qualification continues unchanged. After it passes:

1. Run a bounded end-to-end arithmetic overfit diagnostic in all three conditions, with identical initialization, ordered examples and real-token loss. Include depths 1 and 2. Report exact training-answer accuracy as well as loss.
2. Score fresh, source-disjoint simple programs at depths 1 and 2. Keep them separate from the final experiment's sealed confirmation panel. Declining format loss or memorized training answers alone do not establish arithmetic learning.
3. Record learning curves and actual training/inference cost. If simple arithmetic stays at chance, report that the capability has not been established; do not interpret a pause comparison as evidence against useful computation.
4. Use measured throughput and learning curves to choose the full exposure and budget before the two-seed comparison. Retain the 1024-update first checkpoint and support full-state continuation from the outset, including source offset/order, optimizer, schedule and RNG. Predeclare continuation checkpoints and the schedule before fitting. Keep the late-introduction age fixed in real targets; do not silently move it when a trajectory extends.

Add easier depths 1–2 as diagnostics without replacing the composition endpoints at depths 4, 8 and 12. Longer support is not a promise that a particular token count will be sufficient. Any continuation decision uses TRAIN/DEV diagnostics, not inspected CONFIRM outcomes; all conditions receive aligned exposure. No large run is released by this review.

Two follow-ups remain separate experiments: equal pause exposure at different pretraining ages, and a same-length/same-position dummy-pause control that blocks contextual information from entering pause states. The latter needs explicit causal attention-mask tests. Removing pauses is a useful ablation, but it does not isolate computation from positional structure.

Priority: actual updates and recovery, then learning and fresh simple-program performance. Avoid additional infrastructure work unless a concrete failure blocks those outputs. No 7B scaling, teacher model or RL.
