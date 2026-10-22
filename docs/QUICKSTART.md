# Qualification commands

Use the existing authorized node. Read the protocol and STATUS first. Do not overlap a running attempt or overwrite evidence. No scientific training launcher is supplied.

```bash
source scripts/env.sh
bash scripts/run_checks.sh

python -m scripts.prepare_pause_pretraining --whole-instance \
  --source /home/ubuntu/dario/local/data/uniform-lm-baseline-v1/QUALIFICATION.jsonl \
  --tokenizer /home/ubuntu/dario/local/assets/qwen3-1.7b-base \
  --output /run/user/1000/pause-pretraining-v1

USE_TORCH_XLA=0 python -m train.pause_pretraining_qualify \
  --data /run/user/1000/pause-pretraining-v1/qualification-whole.json \
  --output /run/user/1000/pause-pretraining-v1

python -m scripts.run_pause_whole_instance \
  --root /run/user/1000/pause-pretraining-v1 --attempt 001
```

The data and reference commands create immutable files and intentionally refuse an overwrite. Existing prepared inputs can be reused only when their hashes match. Each native attempt requires a new attempt suffix. All chips must be idle; the supervisor enforces common attempt/time/resource limits and preserves failures.

Qualification sources are retained local inputs from the predecessor. On another machine supply an equivalent 768-window qualification source (JSONL rows with `document_id` and at least 513 Qwen token IDs) and an explicit tokenizer path. Native hardware/stack compatibility must be qualified there.

The output root is temporary memory-backed storage, not a remote backup. Source token IDs, model/optimizer states and compiler outputs stay local. Publish only small numerical receipts after reviewing them.
