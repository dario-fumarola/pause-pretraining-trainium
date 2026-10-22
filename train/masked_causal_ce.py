"""Full-support, T=1 causal CE for explicitly packed supervision.

Positions/labels/bool masks are CPU packing metadata [B,T]; active positions
index TARGET tokens in input_ids [B,S], and must lie in [1,S). The caller
assigns roles: header/prefix/open/reanchor/padding must have neither mask.
Token values never determine masks: legal EOS may equal padding. Precision,
packing, canonical closure and component weights belong to the calling recipe.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class MaskedCausalCE:
    loss: torch.Tensor
    note_close_loss: torch.Tensor
    future_loss: torch.Tensor
    note_close_counts: torch.Tensor
    future_counts: torch.Tensor
    note_close_per_example: torch.Tensor
    future_per_example: torch.Tensor


@dataclass
class ComponentCausalCE:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    counts: dict[str, torch.Tensor]
    per_example: dict[str, torch.Tensor]


def masked_component_causal_ce(backend, input_ids, target_positions, labels,
                               component_masks, component_weights):
    """Average each component per example, then over the fixed batch.

    An absent component contributes zero, including to the batch denominator.
    Inactive metadata slots may use position 0 and label -100. No gradients are
    stopped through note or future CE. Returns differentiable losses and FP32
    per-example counts; no temperature transform or vocabulary mask is applied.
    """
    if (not isinstance(component_masks, dict) or not component_masks
            or not isinstance(component_weights, dict)
            or component_masks.keys() != component_weights.keys()
            or any(not isinstance(name, str) or not name for name in component_masks)):
        raise ValueError('Explicit named component masks and matching weights required')
    names = tuple(component_masks)
    host_masks = tuple(component_masks.values())
    metadata = (target_positions, labels, *host_masks)
    if (input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 2
            or input_ids.dtype != torch.long):
        raise ValueError('Packed input_ids must be nonempty int64 [B,S], S>=2')
    if (any(x.device.type != 'cpu' for x in metadata)
            or target_positions.ndim != 2 or target_positions.shape[0] != input_ids.shape[0]
            or any(x.shape != target_positions.shape for x in metadata)
            or target_positions.dtype != torch.long or labels.dtype != torch.long
            or any(mask.dtype != torch.bool for mask in host_masks)):
        raise ValueError('Positions/labels and bool component masks require aligned CPU [B,T] metadata')
    if any(isinstance(w, bool) or not isinstance(w, (int, float))
           or not math.isfinite(w) or w < 0 for w in component_weights.values()):
        raise ValueError('Component weights must be finite nonnegative numbers')
    active = torch.zeros_like(target_positions, dtype=torch.bool)
    for mask in host_masks:
        if (active & mask).any():
            raise ValueError('Component target masks overlap')
        active |= mask
    if ((target_positions[active] < 1).any()
            or (target_positions[active] >= input_ids.shape[1]).any()
            or (labels[active] < 0).any()
            or (labels[active] >= backend.model.config.vocab_size).any()):
        raise ValueError('Active causal target position or label is out of bounds')
    if any(len(p[m]) != len(p[m].unique()) for p, m in zip(target_positions, active)):
        raise ValueError('Duplicate active target positions')
    if backend.head_chunk < 1:
        raise ValueError('Positive head chunk required')
    # Validate small host metadata before any device work; no XLA scalar reads.
    positions = torch.where(active, target_positions - 1, 0).to(backend.device)
    labels = torch.where(active, labels, 0).to(backend.device)
    masks = torch.stack(host_masks, -1).to(backend.device).float()
    hidden = backend.hidden(input_ids.to(backend.device)[:, :-1])
    selected = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))

    def part(h, y, m):
        with backend.head_amp():
            logits = backend.model.lm_head(h).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                             reduction='none').reshape(y.shape)
        return (ce[:, :, None] * m).sum(1)

    # Preserve a differentiable zero even with T=0; an untied head also receives
    # a zero gradient. Normal nonempty calls use the actual head CE graph below.
    if target_positions.shape[1] == 0:
        zero = hidden.float().sum((1, 2)) * 0.
        zero = zero + backend.model.lm_head.weight.reshape(-1)[0].float() * 0.
        sums = torch.stack([zero] * len(names), -1)
    else:
        sums = torch.zeros((input_ids.shape[0], len(names)), device=backend.device)
        for start in range(0, target_positions.shape[1], backend.head_chunk):
            sl = slice(start, start + backend.head_chunk)
            sums = sums + checkpoint(part, selected[:, sl], labels[:, sl], masks[:, sl],
                                     use_reentrant=True, preserve_rng_state=False)
    # This stack's implicit S64 clamp bound is broken; FP32 counts are exact.
    counts = masks.sum(1)
    means = sums / counts.clamp_min(1.)
    losses = dict(zip(names, means.mean(0).unbind()))
    loss = sum(component_weights[name] * losses[name] for name in names)
    return ComponentCausalCE(loss, losses,
        {name: counts[:, index] for index, name in enumerate(names)},
        {name: means[:, index] for index, name in enumerate(names)})


def masked_causal_ce(backend, input_ids, target_positions, labels,
                     note_close_mask, future_mask, *, note_weight=1., future_weight=1.):
    """Original two-component API, using the same independently normalized core."""
    result = masked_component_causal_ce(backend, input_ids, target_positions, labels,
        dict(note_close=note_close_mask, future=future_mask),
        dict(note_close=note_weight, future=future_weight))
    return MaskedCausalCE(result.loss, result.losses['note_close'], result.losses['future'],
        result.counts['note_close'], result.counts['future'],
        result.per_example['note_close'], result.per_example['future'])
