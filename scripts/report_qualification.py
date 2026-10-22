"""Report actual whole-instance throughput and a proposed, unlaunched budget."""
import argparse
import json
import math
from pathlib import Path
import statistics

from transformers import Qwen3Config
from train.artifacts import atomic_json, sha256_file
from train.pause_pretraining import estimated_flops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = json.loads((a.attempt/'native-report.json').read_text())
    if r['status'] != 'PASS_WHOLE_INSTANCE_NATIVE_QUALIFICATION':
        raise ValueError('Actual completed whole-instance qualification required')
    closures = [json.loads(q.read_text()) for q in sorted(a.root.glob('attempt-*/closure.json'))]
    if any(not q['chip_idle_after'] for q in closures):
        raise ValueError('Every attempt must close with idle devices')
    config = Qwen3Config.from_json_file('configs/qwen3_0.6b_architecture.json')
    v = estimated_flops(config, real_inputs=512, expanded_inputs=512, targets=512)
    e = estimated_flops(config, real_inputs=512, expanded_inputs=576, targets=512)
    samples = {False: [], True: []}
    for arm in ('vanilla', 'early'):
        for row in r['arms'][arm]['steps'][2:]:
            samples[row['pauses']].append(row['maximum_rank_seconds'])
    targets_per_update = 128*512
    mean_tps = {str(k): targets_per_update/statistics.mean(x) for k, x in samples.items()}
    conservative_tps = {str(k): targets_per_update/max(x) for k, x in samples.items()}
    per_fit_targets = 1024*targets_per_update
    # Two vanilla + two early + two half-vanilla/half-pause late fits.
    seconds = 3*per_fit_targets*(1/conservative_tps['False']+1/conservative_tps['True'])
    training_chip_hours = 16*seconds/3600
    actual_qualification = sum(q['assigned_chip_hours'] for q in closures)
    ceiling = math.ceil(1.5*training_chip_hours+actual_qualification+8)
    result = {
        'status': 'MEASURED_BUDGET_PROPOSAL_NO_LARGE_RUN',
        'native_report_sha256': sha256_file(a.attempt/'native-report.json'),
        'actual_qualification_assigned_chip_hours_including_all_attempts': actual_qualification,
        'physical_chips': 16, 'world_size': 64, 'global_batch': 128,
        'measured_later_step_maximum_rank_seconds': {str(k): x for k, x in samples.items()},
        'mean_global_real_targets_per_second': mean_tps,
        'conservative_global_real_targets_per_second': conservative_tps,
        'analytical_training_matmul_FLOPs_per_window': {'vanilla': v['matmul_flops'], 'pauses': e['matmul_flops']},
        'analytical_training_matmul_FLOP_ratio': e['matmul_flops']/v['matmul_flops'],
        'compute_matched_early_completed_updates': int(1024*v['matmul_flops']/e['matmul_flops']),
        'proposed_real_targets_per_fit': per_fit_targets,
        'proposed_six_fit_real_targets': 6*per_fit_targets,
        'proposed_six_fit_training_matmul_FLOPs': 3*1024*128*(v['matmul_flops']+e['matmul_flops']),
        'extrapolated_six_fit_training_node_hours': seconds/3600,
        'extrapolated_six_fit_training_assigned_chip_hours': training_chip_hours,
        'proposed_total_assigned_chip_hour_ceiling': ceiling,
        'proposed_total_full_instance_wall_hour_ceiling': ceiling/16,
        'budget_rule': 'Slowest measured later update per path, six fits at fixed token exposure; 50% contingency, actual qualification including stopped attempts, plus 8 chip-hours for evaluation/export.',
        'limits': [
            'Only four later updates per path; original times and cold compilation remain in the native receipt.',
            'Later updates may contain recompilation; this is a conservative measured estimate, not a claim of peak throughput.',
            'Exact main-run data/splits and export/evaluation/storage paths remain prelaunch work.',
            'No dollar billing rate is assumed. Whole-node invoice depends on allocated wall time and the owner rate.',
            'No scientific outcome or efficiency advantage is established by qualification.'
        ],
        'launch_authorized': False}
    atomic_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
