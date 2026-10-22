"""Small evaluation primitives shared by CPU and native pause experiments."""
import torch
import re
from train.neuron_backend import CausalCache
from train.pause_pretraining import PAUSE_ID, objective, pack


@torch.no_grad()
def score_windows(backend, real_ids, *, pauses, answer_starts=None, answer_lengths=None):
    """Ordinary full-window NLL and optional answer-only NLL, same real targets."""
    from train.masked_causal_ce import masked_component_causal_ce
    b = pack(real_ids, pauses=pauses)
    masks = {'source': torch.ones_like(b.labels, dtype=torch.bool)}
    if answer_starts is not None:
        if answer_lengths is None or len(answer_starts) != len(real_ids) or len(answer_lengths) != len(real_ids):
            raise ValueError('Aligned answer ranges required')
        answer = torch.zeros_like(b.labels, dtype=torch.bool)
        for index, (start, length) in enumerate(zip(answer_starts, answer_lengths)):
            if start < 1 or length < 1 or start+length > real_ids.shape[1]:
                raise ValueError('Answer must lie inside the real source window')
            answer[index, start-1:start-1+length] = True
        # Disjoint masks, then reconstruct full-token-weighted NLL below.
        masks = {'answer': answer, 'other': ~answer}
    r = masked_component_causal_ce(backend, b.input_ids, b.target_positions, b.labels,
        masks, {name: 1. for name in masks})
    full = sum(r.per_example[k]*r.counts[k] for k in masks)/(real_ids.shape[1]-1)
    result = {'full_NLL': full.cpu(), 'predicted_real_tokens': real_ids.shape[1]-1,
              'pause_positions': b.pause_count}
    if answer_starts is not None:
        result['answer_NLL'] = r.per_example['answer'].cpu()
    return result


@torch.no_grad()
def greedy_answer(backend, real_prefix, *, pauses, stop_id, max_new_tokens=4,
                  interval=128, count=16, pause_id=PAUSE_ID):
    """Prefix-only cached generation, including pauses before boundary answers.

    Each row uses a fixed generation cap. EOS and task-specific newline are
    interpreted by the caller. Reserved pause logits are suppressed in every
    arm for generation; likelihood evaluation always uses the full vocabulary.
    """
    if real_prefix.ndim != 2 or real_prefix.shape[1] < 2 or max_new_tokens < 1:
        raise ValueError('Nonempty aligned real prefixes and a fixed cap required')
    batch = pack(real_prefix.cpu(), pauses=pauses, interval=interval, count=count, pause_id=pause_id)
    prefix = batch.input_ids
    if pauses and real_prefix.shape[1] % interval == 0:
        prefix = torch.cat((prefix, torch.full((len(prefix), count), pause_id, dtype=torch.long)), 1)
    device = backend.device
    width = prefix.shape[1]+max_new_tokens+count*((max_new_tokens+interval-1)//interval)
    cache = CausalCache(backend.model.config.num_hidden_layers, width)
    pos = prefix.shape[1]
    logits = backend._next(prefix.to(device), torch.arange(pos, device=device), cache)
    result = []
    alive = torch.ones(len(prefix), dtype=torch.bool, device=device)
    for index in range(max_new_tokens):
        logits[:, pause_id] = -torch.inf
        token = torch.where(alive, logits.argmax(-1), stop_id)
        result.append(token.cpu()); backend.sync()
        alive &= token.ne(stop_id)
        if index+1 == max_new_tokens:
            break
        logits = backend._next(token[:, None], torch.tensor([pos], device=device), cache)
        pos += 1
        real_length = real_prefix.shape[1]+index+1
        if pauses and real_length % interval == 0:
            inputs = torch.full((len(prefix), count), pause_id, dtype=torch.long, device=device)
            logits = backend._next(inputs, torch.arange(pos, pos+count, device=device), cache)
            pos += count
    return torch.stack(result, 1)


@torch.no_grad()
def evaluate_task_batch(backend, tokenizer, rows, *, pauses):
    """Exact accuracy uses only prompt tokens in the model's generation input."""
    starts = [row['task']['answer_start'] for row in rows]
    if not starts or len(set(starts)) != 1:
        raise ValueError('One fixed answer boundary per batch required')
    prefixes = torch.tensor([row['ids'][:starts[0]] for row in rows], dtype=torch.long)
    generated = greedy_answer(backend, prefixes, pauses=pauses,
        stop_id=tokenizer.eos_token_id, max_new_tokens=4)
    outcomes = []
    for row, tokens in zip(rows, generated.tolist()):
        text = tokenizer.decode(tokens, skip_special_tokens=True).split('\n', 1)[0].strip()
        valid = re.fullmatch(r'[0-9]{1,2}', text) is not None and int(text) < 97
        outcomes.append({'correct': bool(valid and int(text) == int(row['task']['answer'])),
            'valid': valid, 'generated_ids': tokens, 'depth': row['task']['depth']})
    return outcomes
