# Project instructions

Current owner direction: a separate random-initialized Qwen3-0.6B pause-pretraining pilot. Read STATUS.md and docs/PAUSE_PRETRAINING_V1.md. Use the whole existing instance (16 Trainium chips / 64 logical cores) for bounded qualification and throughput. Do not launch a large scientific run in this assignment.

Preserve all predecessor results and native model/optimizer/source assets. No teacher model, RL, 7B scaling, social branch, cloud capacity purchases, accelerator upgrades, IAM or network changes. Keep 35 GiB free and 512 GiB available host memory. Include every failed/stopped attempt in resource totals. CPU checks and compilation are distinct from actual hardware qualification. One optimizer owns gradient reduction. Keep data order, real-token exposure and initial parameters matched across conditions; late switching retains optimizer state.

The owner explicitly requests a public new repository and pushing to it. Review code and small numerical reports, then commit and push. Never publish raw source token IDs, weights, optimizer states, caches or secrets. Run bash scripts/run_checks.sh after source changes. Before exit update STATUS.md with active processes/output, actual checks, results and exact next action. Keep the protocol simple; never relabel qualification as scientific success.
