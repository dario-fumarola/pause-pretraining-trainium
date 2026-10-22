"""Causal Qwen trunk and KV cache, copied without numerical changes from the predecessor."""
from contextlib import nullcontext
from typing import Callable
import torch
from transformers.cache_utils import Cache,CacheLayerMixin

class _CausalCacheLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = True

    def __init__(self, holder):
        self.holder = holder
        self.keys = self.values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states, value_states):
        b, h, _, d = key_states.shape
        self.keys = torch.zeros((b, h, self.holder['width'], d),
                                dtype=key_states.dtype, device=key_states.device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys.index_copy_(2, self.holder['positions'], key_states)
        self.values.index_copy_(2, self.holder['positions'], value_states)
        return self.keys, self.values

    def get_seq_length(self):
        # Callers always provide actual position_ids and a complete mask.
        return self.holder['width'] if self.is_initialized else 0

    def get_max_length(self):
        return self.holder['width']

    def get_mask_sizes(self, query_length):
        return self.holder['width'], 0

    def reset(self):
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()


class CausalCache(Cache):
    """Fixed-size cache: tensor-valued positions prevent per-position graphs."""
    def __init__(self, layers: int, width: int):
        self.holder = {'width': width, 'positions': None}
        super().__init__(layers=[_CausalCacheLayer(self.holder) for _ in range(layers)])

    def set_positions(self, positions):
        self.holder['positions'] = positions


def causal_mask(positions, width):
    """Explicit additive causal mask; no host extraction of position values."""
    keys = torch.arange(width, device=positions.device)
    return torch.where(keys[None, :] <= positions[:, None], 0.0, -1e9)[None, None]


class NeuronCausalBackend:
    def __init__(self, model, *, amp: Callable = nullcontext,
                 sync: Callable = lambda: None, head_chunk: int = 32,
                 checkpoint_trunk: bool = True, banned_ids: tuple[int, ...] = (),
                 stop_check_interval: int = 32, head_amp: Callable | None = None):
        self.model = model
        self.amp = amp
        self.head_amp = amp if head_amp is None else head_amp
        self.sync = sync
        self.head_chunk = head_chunk
        self.checkpoint_trunk = checkpoint_trunk
        self.banned_ids = banned_ids
        if stop_check_interval < 1:
            raise ValueError('Positive stop-check interval required')
        self.stop_check_interval = stop_check_interval
        if model.config.model_type != 'qwen3' or any(t != 'full_attention' for t in model.config.layer_types):
            raise ValueError('This adapter supports ordinary causal Qwen3 only')
        self.model.eval()  # Deterministic dropout, without disabling gradients.


    @property
    def device(self):
        return next(self.model.parameters()).device


    def hidden(self, x):
        positions = torch.arange(x.shape[-1], device=x.device)
        mask = causal_mask(positions, x.shape[-1])
        if self.checkpoint_trunk and torch.is_grad_enabled():
            # HF's checkpointing requires train(); explicitly checkpoint here
            # so deterministic eval() and memory-bounded replay coexist.
            from torch.utils.checkpoint import checkpoint
            trunk = self.model.model
            with self.amp():
                h = trunk.embed_tokens(x)
                pe = trunk.rotary_emb(h, positions[None])
            for layer in trunk.layers:
                def block(h, cos, sin, layer=layer):
                    with self.amp():
                        return layer(h, attention_mask=mask, position_ids=positions[None],
                                     position_embeddings=(cos, sin), use_cache=False)
                h = checkpoint(block, h, pe[0], pe[1], use_reentrant=True,
                               preserve_rng_state=False)
            with self.amp():
                return trunk.norm(h)
        with self.amp():
            return self.model.model(input_ids=x, position_ids=positions[None],
                attention_mask={'full_attention': mask},
                use_cache=False).last_hidden_state


    def _next(self, tokens, positions, cache):
        cache.set_positions(positions)
        with self.amp():
            h = self.model.model(input_ids=tokens, position_ids=positions[None],
                attention_mask={'full_attention': causal_mask(positions, cache.holder['width'])},
                past_key_values=cache, use_cache=True).last_hidden_state
        with self.head_amp():
            return self.model.lm_head(h[:, -1]).float()


