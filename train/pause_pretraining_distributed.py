"""Whole-instance, 64-rank qualification. Six updates per condition, never science."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from train.neuron_backend import NeuronCausalBackend
from train.pause_pretraining import EXPERIMENT, PAUSE_ID, enabled, learning_rate, objective, pack, source_digest
from train.pause_pretraining_qualify import cpu_tree, model_digest, model_from_config
from train.runtime_utils import cast_parameters
from train.runtime_utils import optimizer_digest
from train.artifacts import atomic_json, canonical_hash, sha256_file
from train.runtime_utils import sampled_fingerprint


def source_indices(step, rank, micro, *, world=64):
    if not (0 <= step < 6 and 0 <= rank < world and 0 <= micro < 2):
        raise ValueError('Bounded whole-instance qualification index required')
    return step*(2*world)+micro*world+rank


def bounded_compiler(root):
    """Process-local scheduling only: four compilers, original flags/NEFFs.

    Per-entry cache locks already prevent duplicate compilation. This bounds
    aggregate host memory when different rank-specific graphs compile at once.
    """
    import libneuronxla.neuron_cc_wrapper as wrapper
    original = wrapper.compile_cache_entry
    locks = root/'compiler-slots'; locks.mkdir(exist_ok=True)
    def compile_limited(*args, **kwargs):
        handles = [(locks/str(i)).open('a+b') for i in range(4)]
        selected = None
        try:
            while selected is None:
                for handle in handles:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        selected = handle
                        break
                    except BlockingIOError:
                        pass
                if selected is None:
                    time.sleep(.25)
            kwargs['work_dir'] = str(root/'compile-work')
            return original(*args, **kwargs)
        finally:
            for handle in handles:
                handle.close()
    wrapper.compile_cache_entry = compile_limited


def worker(_index, a):
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa: F401
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
    from train.runtime_compat import configure_runtime, restore_syncfree_device_state
    from train.pause_pretraining_evaluate import greedy_answer

    configure_runtime(); bounded_compiler(a.output.parent)
    if not hasattr(torch, 'xla'):
        torch._register_device_module('xla', torch_xla)
    device = torch_xla.device(); rank, world = xr.global_ordinal(), xr.world_size()
    if world != 64:
        raise ValueError('The owner-requested full instance requires 64 logical cores')
    dist.init_process_group('xla', init_method='xla://')
    sync = lambda: torch_xla.sync(wait=True)
    def consensus(ok, message):
        failed = xm.all_reduce(xm.REDUCE_MAX, torch.tensor(float(not ok), device=device))
        sync()
        if bool(failed.cpu()):
            raise RuntimeError(message)
    rows = json.loads(a.data.read_text())
    ref = json.loads((a.output/'cpu-reference-whole.json').read_text())
    consensus(ref['data_sha256'] == sha256_file(a.data) and ref['config_sha256'] == sha256_file(a.config),
              'Reference/data/config identity changed')
    report = {'experiment_id': EXPERIMENT, 'qualification_id': 'whole-instance-v1',
        'world_size': world, 'physical_chips': 16, 'global_batch': 128,
        'micro_batch': 1, 'accumulation': 2, 'head_chunk': 128,
        'data_sha256': sha256_file(a.data), 'ordered_source_digest': source_digest(rows),
        'parameter_count': ref['parameter_count'], 'arms': {}}
    for arm in ('vanilla', 'early', 'late'):
        model = model_from_config(a.config); cast_parameters(model)
        consensus(sampled_fingerprint(model) == ref['initial_fingerprint'], 'Random initialization differs')
        model.to(device); model.tie_weights()
        backend = NeuronCausalBackend(model, sync=sync, head_chunk=128)
        optimizer = ZeroRedundancyOptimizer(model.parameters(), AdamW,
            optimizer_dtype=torch.float32, grad_clipping=True, max_norm=1.,
            save_master_weights=True, lr=3e-4, betas=(.9, .95), eps=1e-8,
            weight_decay=.1, use_grad_acc_hook=False, higher_cc_precision=False)
        reductions = [0]; original_reduce = optimizer._reduce_gradients
        def counted_reduce(*args, **kwargs):
            reductions[0] += 1
            return original_reduce(*args, **kwargs)
        optimizer._reduce_gradients = counted_reduce
        found_inf = torch.zeros((), device=device); sync()
        active = arm == 'early'
        batch = pack(torch.tensor([rows[rank % 2]['ids']]), pauses=active)
        with torch.no_grad():
            ce = objective(backend, batch); sync(); initial_loss = float(ce.loss.cpu())
        error = abs(initial_loss-ref['references'][str(active)]['per_example'][rank % 2])
        consensus(error <= .03 and math.isfinite(initial_loss), 'Initial CPU/native CE parity failed')
        errors = xm.all_gather(torch.tensor([error], device=device)); sync()
        record = {'maximum_initial_CPU_CE_error': float(errors.max().cpu()),
                  'steps': [], 'inference': {}, 'cached_generation': {}}
        del ce

        def update(step, replay=False):
            active = enabled(arm, step, 3)
            rate = learning_rate(step, total=6, warmup=2)
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            xm.rendezvous(f'{arm}-{step}-{replay}-start')
            start = time.monotonic(); loss = 0.; before_reductions = reductions[0]
            master = optimizer.base_optimizer.param_groups[0]['params'][0]
            before = master.reshape(-1)[:32].detach().cpu().clone()
            for micro in range(2):
                index = source_indices(step, rank, micro)
                batch = pack(torch.tensor([rows[index]['ids']]), pauses=active)
                ce = objective(backend, batch)
                (ce.loss/2).backward(); sync()
                loss += float(ce.loss.detach().cpu())/2
            optimizer.step(found_inf=found_inf); sync()
            norm = float(optimizer.grad_norm.cpu())
            after = master.reshape(-1)[:32].detach().cpu()
            changed = not torch.equal(before, after)
            masters = [p for group in optimizer.base_optimizer.param_groups for p in group['params']]
            actual_lr = [float(g['lr']) for g in optimizer.base_optimizer.param_groups]
            ok = (reductions[0]-before_reductions == 1 and changed and all(x == rate for x in actual_lr)
                and all(p.dtype == torch.float32 for p in masters)
                and math.isfinite(loss) and math.isfinite(norm) and norm > 0)
            consensus(ok, 'Native loss/gradient/reducer/LR/FP32 update failed')
            seconds = time.monotonic()-start
            times = xm.all_gather(torch.tensor([seconds], device=device)); sync()
            losses = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(loss/world, device=device)); sync()
            entry = {'step': step, 'pauses': active, 'mean_loss': float(losses.cpu()),
                'gradient_norm': norm, 'maximum_rank_seconds': float(times.max().cpu()),
                'actual_LR': rate, 'all_ranks_FP32_master_sample_changed': True,
                'one_gradient_reduction_per_rank': True, 'predicted_real_tokens': 128*512,
                'real_source_tokens': 128*513, 'trunk_input_positions': 128*(576 if active else 512),
                'replay': replay}
            if rank == 0:
                print(json.dumps({'arm': arm, **entry}), flush=True)
            return entry

        checkpoint = a.output/'late-before-switch'
        if rank == 0:
            checkpoint.mkdir(exist_ok=True)
        xm.rendezvous(f'{arm}-checkpoint-directory')
        for step in range(6):
            record['steps'].append(update(step))
            if arm == 'late' and step == 2:
                optimizer.zero_grad(set_to_none=True); sync()
                if rank == 0:
                    torch.save(cpu_tree(model.state_dict()), checkpoint/'model.pt')
                payload = {'optimizer': cpu_tree(optimizer.state_dict()), 'next_step': 3,
                    'CPU_rng': torch.get_rng_state(), 'device_rng': xm.get_rng_state(device),
                    'data_sha256': sha256_file(a.data), 'world': world, 'rank': rank}
                torch.save(payload, checkpoint/f'rank-{rank:03d}.pt'); del payload
                xm.rendezvous('complete-late-before-switch')
                if rank == 0:
                    atomic_json(checkpoint/'COMPLETE.json', {
                        'world_size': world, 'next_step': 3,
                        'files': {q.name: sha256_file(q) for q in checkpoint.glob('*.pt')}})
                xm.rendezvous('late-complete-manifest')

        # Replication agreement: every rank hashes its full terminal BF16 model.
        terminal = canonical_hash(model_digest(model))
        digest = torch.tensor(list(bytes.fromhex(terminal)), dtype=torch.int32, device=device)
        gathered = xm.all_gather(digest).reshape(world, -1); sync()
        all_equal = bool((gathered.cpu() == gathered.cpu()[0]).all())
        consensus(all_equal, 'Full terminal model replicas differ')
        record['full_terminal_model_replicas_equal'] = True
        record['terminal_model_digest'] = terminal
        for active in (False, True):
            batch = pack(torch.tensor([rows[rank]['ids']]), pauses=active)
            times = []
            with torch.no_grad():
                for _ in range(4):
                    xm.rendezvous(f'{arm}-score-{active}-{_}')
                    start = time.monotonic(); ce = objective(backend, batch); sync()
                    times.append(time.monotonic()-start)
            local = torch.tensor(times, device=device)
            maximum = xm.all_reduce(xm.REDUCE_MAX, local); sync()
            record['inference'][str(active)] = {'maximum_rank_seconds': maximum.cpu().tolist(),
                'predicted_real_tokens_per_call': world*512}
        prefix = torch.tensor([rows[rank]['ids'][:128]])
        for active in (False, True):
            times = []
            for repeat in range(4):
                xm.rendezvous(f'{arm}-cached-{active}-{repeat}')
                start = time.monotonic()
                result = greedy_answer(backend, prefix, pauses=active, stop_id=151643, max_new_tokens=4)
                sync(); times.append(time.monotonic()-start)
            maximum = xm.all_reduce(xm.REDUCE_MAX, torch.tensor(times, device=device)); sync()
            record['cached_generation'][str(active)] = {'maximum_rank_seconds': maximum.cpu().tolist(),
                'requests_per_call': world, 'prefix_real_tokens': 128, 'decode_cap': 4}
        if arm == 'late':
            expected_model = model_digest(model)
            expected_optimizer = optimizer_digest(cpu_tree(optimizer.state_dict()))
            saved_model = torch.load(checkpoint/'model.pt', map_location='cpu', weights_only=False)
            saved = torch.load(checkpoint/f'rank-{rank:03d}.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(saved_model); optimizer.load_state_dict(saved['optimizer'])
            restore_syncfree_device_state(optimizer); sync()
            torch.set_rng_state(saved['CPU_rng']); xm.set_rng_state(saved['device_rng'], device)
            del saved, saved_model
            replay = [update(step, replay=True) for step in range(3, 6)]
            model_equal = model_digest(model) == expected_model
            optimizer_equal = optimizer_digest(cpu_tree(optimizer.state_dict())) == expected_optimizer
            consensus(model_equal and optimizer_equal, 'Full distributed replay differs')
            record['recovery'] = {'every_rank_full_model_bitwise_equal': True,
                'every_FP32_master_moment_counter_bitwise_equal': True, 'replay': replay}
        report['arms'][arm] = record
        if rank == 0:
            atomic_json(a.output/f'native-progress-{arm}.json', report)
        optimizer._reduce_gradients = original_reduce
        del optimizer, original_reduce, model, backend, ce
        gc.collect(); sync(); xm.rendezvous(f'{arm}-closed')
    if rank == 0:
        report['status'] = 'PASS_WHOLE_INSTANCE_NATIVE_QUALIFICATION'
        atomic_json(a.output/'native-report.json', report)
        print(report['status'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/qwen3_0.6b_architecture.json'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.data.read_text())
    if len(rows) != 768 or any(len(r['ids']) != 513 for r in rows):
        raise ValueError('Only 768 fixed qualification windows are accepted')
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(a,), nprocs=None)


if __name__ == '__main__':
    main()
