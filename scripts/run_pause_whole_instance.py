"""Bounded qualification on all existing 16 chips / 64 logical Neuron cores."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from train.artifacts import atomic_json, sha256_file, utc_now


def available():
    with open('/proc/meminfo') as f:
        return int(next(line for line in f if line.startswith('MemAvailable:')).split()[1])*1024


def inventory():
    return json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls','-j']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--attempt', required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo/'configs/pause_pretraining_v1.json').read_text())['whole_instance_qualification']
    closures = [json.loads(q.read_text()) for q in a.root.glob('attempt-*/closure.json')]
    launches = list(a.root.glob('attempt-*/launch.json'))
    used = sum(q['assigned_chip_hours'] for q in closures)
    if len(closures) != len(launches) or len(launches) >= cfg['maximum_total_attempts_including_single_rank']:
        raise RuntimeError('An attempt is still open or the common attempt bound was reached')
    limit = min(cfg['maximum_wall_seconds_per_attempt'], (cfg['maximum_total_assigned_chip_hours']-used)*3600/16)
    if limit < 60:
        raise RuntimeError('Insufficient remaining full-instance qualification budget')
    before = inventory()
    if len(before) != 16 or any(q['neuron_processes'] or q['logical_neuroncore_config'] != 2 for q in before):
        raise RuntimeError('All existing chips must be idle in the installed logical configuration')
    if available() < 768*2**30:
        raise RuntimeError('Insufficient full-instance staging headroom')
    roots = [repo, a.root]
    if min(shutil.disk_usage(q).free for q in roots) < 35*2**30:
        raise RuntimeError('Storage floor failed')
    stage = a.root/f'attempt-whole-{a.attempt}'; stage.mkdir(exist_ok=False)
    for name in ('cpu-reference-whole.json',):
        shutil.copy2(a.root/name, stage/name)
    env = os.environ.copy()
    for key in ('XLA_USE_BF16', 'XLA_DOWNCAST_BF16', 'NEURON_NUM_RECENT_MODELS_TO_KEEP',
        'TORCHELASTIC_RUN_ID', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
        'NEURON_PJRT_PROCESS_INDEX', 'NEURON_PJRT_PROCESSES_NUM_DEVICES', 'NEURON_PT_MPMD_MODE'):
        env.pop(key, None)
    tmp = a.root/'tmp'; tmp.mkdir(exist_ok=True)
    env.update(PJRT_DEVICE='NEURON', NEURON_LOGICAL_NC_CONFIG='2',
        NEURONCORE_NUM_DEVICES='64', NEURON_RT_VISIBLE_CORES='0-63',
        NEURON_RT_CONTIGUOUS_SCRATCHPAD='0', NEURON_RT_EXEC_TIMEOUT='1800', MASTER_ADDR='127.0.0.1',
        MASTER_PORT='29780', NEURON_RT_ROOT_COMM_ID='127.0.0.1:29781',
        NEURON_CC_FLAGS='--model-type=transformer --auto-cast=none',
        NEURON_COMPILE_CACHE_URL=str(a.root/'cache-whole'), XLA_COMPILATION_CACHE_SIZE='2',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', USE_TORCH_XLA='0',
        TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1')
    for port in (29780, 29781):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    command = [sys.executable, '-u', '-m', 'train.pause_pretraining_distributed',
        '--data', str(a.root/'qualification-whole.json'), '--output', str(stage)]
    names = ['train/pause_pretraining.py', 'train/pause_pretraining_qualify.py',
        'train/pause_pretraining_distributed.py', 'train/pause_pretraining_evaluate.py',
        'train/neuron_backend.py', 'train/masked_causal_ce.py', 'train/runtime_compat.py',
        'scripts/run_pause_whole_instance.py', 'train/artifacts.py', 'train/runtime_utils.py',
        'configs/pause_pretraining_v1.json', 'configs/qwen3_0.6b_architecture.json']
    for name in names:
        dest = stage/'sources'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, dest)
    start = time.monotonic(); observed = set(); failure = None; minimum_memory = available()
    with (stage/'run.log').open('xb') as log, (stage/'resources.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(stage/'launch.json', {'utc': utc_now(), 'pid': child.pid, 'command': command,
            'physical_chips': list(range(16)), 'visible_cores': '0-63',
            'world_size': 64, 'wall_limit_seconds': limit, 'already_used_chip_hours': used,
            'maximum_total_assigned_chip_hours': cfg['maximum_total_assigned_chip_hours'],
            'source_files': {name: sha256_file(repo/name) for name in names}})
        try:
            ticks = 0
            while child.poll() is None:
                memory = available(); minimum_memory = min(minimum_memory, memory)
                free = {str(q): shutil.disk_usage(q).free for q in roots}
                elapsed = time.monotonic()-start
                samples.write(json.dumps({'utc': utc_now(), 'available_bytes': memory,
                    'free_bytes': free, 'seconds': elapsed})+'\n'); samples.flush()
                if ticks % 5 == 0:
                    topology = inventory()
                    occupied = [q for q in topology if q['neuron_processes']]
                    observed.update(q['neuron_device'] for q in occupied)
                    for chip in occupied:
                        for proc in chip['neuron_processes']:
                            if os.getsid(proc['pid']) != child.pid:
                                raise RuntimeError('An unowned process appeared on the instance')
                    if len(occupied) == 16 and not (stage/'all-16-chips-observed.json').exists():
                        atomic_json(stage/'all-16-chips-observed.json', topology)
                if memory < 512*2**30 or min(free.values()) < 35*2**30 or elapsed >= limit-15:
                    raise RuntimeError('Declared host-memory/storage/time boundary reached')
                ticks += 1; time.sleep(2)
        except BaseException as error:
            failure = repr(error)
            os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); code = child.wait()
    # Workers can outlive an xmp parent failure; close only this owned session.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(10):
        after = inventory()
        if all(not q['neuron_processes'] for q in after):
            break
        time.sleep(1)
    idle = all(not q['neuron_processes'] for q in after)
    elapsed = time.monotonic()-start
    closure = {'utc': utc_now(), 'exit_code': code, 'failure': failure,
        'wall_seconds': elapsed, 'assigned_chip_hours': 16*elapsed/3600,
        'total_qualification_assigned_chip_hours': used+16*elapsed/3600,
        'minimum_host_available_GiB': minimum_memory/2**30,
        'observed_physical_chips': sorted(observed), 'all_16_observed': len(observed) == 16,
        'chip_idle_after': idle}
    atomic_json(stage/'closure.json', closure); print(json.dumps(closure), flush=True)
    if code or failure or not idle or len(observed) != 16:
        raise RuntimeError('Whole-instance qualification failed; evidence and costs retained')


if __name__ == '__main__':
    main()
