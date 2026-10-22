"""Bounded qualification in four groups spanning all 16 existing chips."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from train.artifacts import atomic_json, sha256_file, utc_now


def available():
    with open('/proc/meminfo') as f:
        return int(next(s for s in f if s.startswith('MemAvailable:')).split()[1])*1024


def inventory():
    return json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls', '-j']))


def stop(children):
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


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
        raise RuntimeError('Previous attempt still open or common attempt limit reached')
    limit = min(cfg['maximum_wall_seconds_per_attempt'], (cfg['maximum_total_assigned_chip_hours']-used)*3600/16)
    before = inventory()
    if limit < 60 or len(before) != 16 or any(q['neuron_processes'] for q in before):
        raise RuntimeError('Insufficient remaining budget or not all chips idle')
    if available() < 768*2**30 or min(shutil.disk_usage(q).free for q in (repo, a.root)) < 35*2**30:
        raise RuntimeError('Prelaunch resource floor failed')
    stage = a.root/f'attempt-whole-{a.attempt}'; stage.mkdir(exist_ok=False)
    names = sorted([str(q.relative_to(repo)) for q in (repo/'train').glob('*.py')])
    names += ['scripts/run_pause_whole_instance.py', 'configs/pause_pretraining_v1.json', 'configs/qwen3_0.6b_architecture.json']
    for name in names:
        dest = stage/'sources'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, dest)
    tmp = a.root/'tmp'; tmp.mkdir(exist_ok=True)
    children, logs, jobs = [], [], []
    start = time.monotonic(); failure = None; observed = set(); minimum_memory = available()
    codes = []
    try:
        for group in range(4):
            output = stage/f'group-{group}'; output.mkdir()
            shutil.copy2(a.root/'cpu-reference-whole.json', output/'cpu-reference-whole.json')
            env = os.environ.copy()
            for key in ('XLA_USE_BF16', 'XLA_DOWNCAST_BF16', 'NEURON_NUM_RECENT_MODELS_TO_KEEP',
                'TORCHELASTIC_RUN_ID', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                'NEURON_PJRT_PROCESS_INDEX', 'NEURON_PJRT_PROCESSES_NUM_DEVICES', 'NEURON_PT_MPMD_MODE'):
                env.pop(key, None)
            first = group*16; port = 29800+group*2
            env.update(PJRT_DEVICE='NEURON', NEURON_LOGICAL_NC_CONFIG='2', NEURONCORE_NUM_DEVICES='16',
                NEURON_RT_VISIBLE_CORES=f'{first}-{first+15}', NEURON_RT_CONTIGUOUS_SCRATCHPAD='0',
                NEURON_RT_EXEC_TIMEOUT='1800', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}', NEURON_CC_FLAGS='--model-type=transformer --auto-cast=none',
                NEURON_COMPILE_CACHE_URL=str(a.root/'cache-whole'), XLA_COMPILATION_CACHE_SIZE='2',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', USE_TORCH_XLA='0', TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, '-u', '-m', 'train.pause_pretraining_distributed',
                '--data', str(a.root/'qualification-whole.json'), '--output', str(output), '--compile-root', str(a.root)]
            log = (output/'run.log').open('xb'); logs.append(log)
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            jobs.append({'group': group, 'pid': child.pid, 'visible_cores': env['NEURON_RT_VISIBLE_CORES'], 'command': command})
        atomic_json(stage/'launch.json', {'utc': utc_now(), 'jobs': jobs, 'physical_chips': list(range(16)),
            'layout': 'four identical 16-rank hardware qualification copies, not independent science replications',
            'wall_limit_seconds': limit, 'already_used_chip_hours': used,
            'source_files': {name: sha256_file(repo/name) for name in names}})
        with (stage/'resources.jsonl').open('x') as samples:
            ticks = 0
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('A qualification group failed; close all owned groups')
                memory = available(); minimum_memory = min(minimum_memory, memory)
                free = {str(q): shutil.disk_usage(q).free for q in (repo, a.root)}
                elapsed = time.monotonic()-start
                samples.write(json.dumps({'utc': utc_now(), 'available_bytes': memory,
                    'free_bytes': free, 'seconds': elapsed})+'\n'); samples.flush()
                if ticks % 5 == 0:
                    topology = inventory(); sessions = {child.pid for child in children}
                    occupied = [q for q in topology if q['neuron_processes']]
                    observed.update(q['neuron_device'] for q in occupied)
                    if any(os.getsid(proc['pid']) not in sessions for q in occupied for proc in q['neuron_processes']):
                        raise RuntimeError('Unowned device process detected')
                    if len(occupied) == 16 and not (stage/'all-16-chips-observed.json').exists():
                        atomic_json(stage/'all-16-chips-observed.json', topology)
                if memory < 512*2**30 or min(free.values()) < 35*2**30 or elapsed >= limit-15:
                    raise RuntimeError('Host-memory/storage/time boundary reached')
                ticks += 1; time.sleep(2)
        codes = [child.wait() for child in children]
    except BaseException as error:
        failure = repr(error)
    finally:
        stop(children)
        for log in logs:
            log.close()
    for _ in range(10):
        after = inventory()
        if all(not q['neuron_processes'] for q in after):
            break
        time.sleep(1)
    idle = all(not q['neuron_processes'] for q in after)
    elapsed = time.monotonic()-start
    closure = {'utc': utc_now(), 'exit_codes': codes or [child.returncode for child in children], 'failure': failure,
        'wall_seconds': elapsed, 'assigned_chip_hours': 16*elapsed/3600,
        'total_qualification_assigned_chip_hours': used+16*elapsed/3600,
        'minimum_host_available_GiB': minimum_memory/2**30, 'observed_physical_chips': sorted(observed),
        'all_16_observed': len(observed) == 16, 'chip_idle_after': idle}
    atomic_json(stage/'closure.json', closure); print(json.dumps(closure), flush=True)
    if failure or any(closure['exit_codes']) or not idle or len(observed) != 16:
        raise RuntimeError('Qualification did not close successfully; all evidence and costs retained')
    reports = [json.loads((stage/f'group-{g}'/'native-report.json').read_text()) for g in range(4)]
    if any(r['status'] != 'PASS_DISTRIBUTED_GROUP_NATIVE_QUALIFICATION' for r in reports):
        raise RuntimeError('Every group must pass all numerical/recovery checks')
    atomic_json(stage/'native-report.json', {'status': 'PASS_WHOLE_INSTANCE_NATIVE_QUALIFICATION',
        'physical_chips': 16, 'logical_cores': 64, 'groups': reports,
        'limits': 'Four identical hardware copies, not independent scientific training seeds or new experimental conditions.'})


if __name__ == '__main__':
    main()
