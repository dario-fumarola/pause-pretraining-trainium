"""Source-aligned, teacher-free pause pretraining. No existing recipe is changed.

The real token after a pause is predicted FROM THE LAST PAUSE, not from the
preceding real token. Pause inputs have no prediction targets of their own.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random

import torch
from train.masked_causal_ce import masked_component_causal_ce

EXPERIMENT = 'pause-pretraining-qwen3-0.6b-v1'
ARMS = ('vanilla', 'early', 'late')
PAUSE_ID = 151935  # Existing, tokenizer-undefined row; no vocabulary resize.


def enabled(arm, step, late_start):
    if arm not in ARMS or type(step) is not int or step < 0 or type(late_start) is not int or late_start < 1:
        raise ValueError('Valid arm and zero-based optimizer step required')
    return arm == 'early' or arm == 'late' and step >= late_start


@dataclass
class PauseBatch:
    input_ids: torch.Tensor
    target_positions: torch.Tensor
    labels: torch.Tensor
    real_positions: torch.Tensor
    pause_count: int


def pack(real_ids, *, pauses, interval=128, count=16, pause_id=PAUSE_ID):
    """Insert only before a real successor; never truncate any real token.

    Windows include one initial context token, followed by equally many targets
    in all arms. Position IDs use the expanded causal sequence. There is no
    padding: every supplied real token is a source token, including any EOS.
    """
    if (real_ids.device.type != 'cpu' or real_ids.dtype != torch.long
            or real_ids.ndim != 2 or real_ids.shape[0] < 1 or real_ids.shape[1] < 2
            or type(pauses) is not bool or type(interval) is not int or interval < 1
            or type(count) is not int or count < 1 or type(pause_id) is not int or pause_id < 0
            or (real_ids < 0).any() or (real_ids == pause_id).any()):
        raise ValueError('Nonempty CPU int64 real windows excluding the reserved pause ID required')
    pieces, positions, offset = [], [], 0
    for start in range(0, real_ids.shape[1], interval):
        block = real_ids[:, start:start+interval]
        if start and pauses:
            pieces.append(torch.full((real_ids.shape[0], count), pause_id, dtype=torch.long))
            offset += count
        pieces.append(block)
        positions.extend(range(offset, offset+block.shape[1]))
        offset += block.shape[1]
    x = torch.cat(pieces, 1)
    rp = torch.tensor(positions, dtype=torch.long)
    return PauseBatch(x, rp[None, 1:].expand(real_ids.shape[0], -1).clone(),
                      real_ids[:, 1:].clone(), rp, x.shape[1]-real_ids.shape[1])


def objective(backend, batch):
    return masked_component_causal_ce(backend, batch.input_ids,
        batch.target_positions, batch.labels,
        {'source': torch.ones_like(batch.labels, dtype=torch.bool)}, {'source': 1.})


def source_digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()+b'\n')
    return h.hexdigest()


def learning_rate(step, *, total, warmup, peak=3e-4, floor=.1):
    if not (0 <= step < total and 0 < warmup < total and 0 < peak <= 1e-3 and 0 <= floor <= 1):
        raise ValueError('Bounded pretraining schedule required')
    if step < warmup:
        return peak*(step+1)/warmup
    progress = (step+1-warmup)/(total-warmup)
    return peak*(floor+(1-floor)*.5*(1+math.cos(math.pi*progress)))


def arithmetic_program(seed, split, index, depth):
    """Disjoint RNG namespaces, final answer only; no intermediate-state labels."""
    if split not in ('qualification', 'train', 'dev', 'confirm') or depth not in (4, 8, 12, 16):
        raise ValueError('Declared task split and depth required')
    key = hashlib.sha256(f'{seed}:{split}:{index}:{depth}'.encode()).digest()
    rng = random.Random(int.from_bytes(key, 'big'))
    initial = rng.randrange(97)
    operations = [(rng.choice((2, 3, 5, 7)), rng.randrange(97)) for _ in range(depth)]
    x = initial
    for a, b in operations:
        x = (a*x+b) % 97
    prompt = f'Mod 97. Start {initial}. For each (a,b), set x=(a*x+b) mod 97: '
    prompt += ' '.join(f'({a},{b})' for a, b in operations)+'\nAnswer:'
    return {'key': key.hex(), 'initial': initial, 'operations': operations,
            'depth': depth, 'prompt': prompt, 'answer': str(x)}


def task_window(tokenizer, program, tail_ids, *, width=513, boundary=128):
    """Left neutral newlines align the answer at a declared real-token boundary.

    Alignment is computed from the prompt ONLY. Answer length/content never
    affects prompt selection, delay placement, or inference inputs.
    """
    prompt = tokenizer.encode(program['prompt'], add_special_tokens=False)
    if len(prompt) > boundary or len(tail_ids) < width:
        raise ValueError('Task prompt exceeds the fixed boundary or missing ordinary tail')
    neutral = tokenizer.encode('\n', add_special_tokens=False)
    if len(neutral) != 1:
        raise ValueError('One-token neutral alignment required')
    context = neutral*(boundary-len(prompt))+prompt
    answer = tokenizer.encode(program['answer']+'\n', add_special_tokens=False)
    completion = answer+[tokenizer.eos_token_id]
    if boundary+len(completion) > width:
        raise ValueError('Task completion exceeds the window')
    ids = context+completion+list(tail_ids[:width-boundary-len(completion)])
    return ids, {'answer_start': boundary, 'answer_tokens': len(answer)-1,
                 'prompt_tokens': len(prompt), 'program_key': program['key'],
                 'depth': program['depth'], 'answer': program['answer']}


def estimated_flops(config, *, real_inputs, expanded_inputs, targets, batch=1, training=True):
    """Forward matmul FLOPs; excludes norms/softmax/elementwise work.

    Counts dense causal attention's full square, not a triangular sparse kernel.
    Training assumes 3x forward plus one rematerialized trunk/head forward = 4x.
    This is an analytical estimate, never hardware-achieved FLOPs.
    """
    h, i, l = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    q = config.num_attention_heads*config.head_dim
    kv = config.num_key_value_heads*config.head_dim
    s = expanded_inputs
    projections = 2*s*(h*q+2*h*kv+q*h+3*h*i)*l
    attention = 4*s*s*q*l
    head = 2*targets*h*config.vocab_size
    return {'matmul_flops': batch*(projections+attention+head)*(4 if training else 1),
            'real_inputs': real_inputs*batch, 'expanded_inputs': s*batch,
            'targets': targets*batch, 'recompute_multiplier': 4 if training else 1}
