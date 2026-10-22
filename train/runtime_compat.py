"""Narrow compatibility fixes for the exact installed Neuron runtime.

These modify only process-local Python callables. Installed files and compiler
outputs remain intact. Unknown library versions or source contents fail closed.
"""

from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
from typing import Any


QUALIFIED_LIBNEURONXLA_VERSION = "2.2.17544.0+fb9962bf"
QUALIFIED_COMPILER_WRAPPER_SHA256 = "2068ce8be8a95fdb4e8a56c6927ef3357adb1e6242ce729ed40d7e02bd3d9036"


def _supply_wrapped_output_path(function):
    signature = inspect.signature(function)
    if "output_wrapped_neff" not in signature.parameters:
        raise RuntimeError("Installed compiler wrapper has an unexpected signature")

    @wraps(function)
    def compile_cache_entry(*arguments, **keywords):
        # Bind rather than indexing kwargs, so positional optional args work too.
        bound = signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        if bound.arguments["output_wrapped_neff"] is None:
            output = Path(bound.arguments["output_neff"])
            bound.arguments["output_wrapped_neff"] = str(output.with_name("wrapped_neff.hlo"))
        return function(*bound.args, **bound.kwargs)

    return compile_cache_entry


def configure_runtime() -> dict[str, Any]:
    """Supply a missing optional compiler artifact path for the pinned stack.

    The installed ordinary compile caller omits output_wrapped_neff, but the
    compiler can emit wrapped_neff.hlo for optimizer graphs. The cache writer
    then tries shutil.copyfile(..., None), after successfully compiling. Passing
    a temporary path beside output_neff fixes that Python bookkeeping error;
    compiler arguments, NEFF bytes and cache keys are unaffected. Exceptions
    from compilation still propagate. Repeated configuration is idempotent.
    """
    import libneuronxla.neuron_cc_wrapper as wrapper
    from libneuronxla.version import __version__

    if __version__ != QUALIFIED_LIBNEURONXLA_VERSION:
        raise RuntimeError(f"Unqualified libneuronxla version: {__version__}")
    source = Path(wrapper.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != QUALIFIED_COMPILER_WRAPPER_SHA256:
        raise RuntimeError(f"Unqualified compiler-wrapper source hash: {digest}")
    if not getattr(wrapper.compile_cache_entry, "_dario_wrapped_output_compat", False):
        patched = _supply_wrapped_output_path(wrapper.compile_cache_entry)
        patched._dario_wrapped_output_compat = True
        wrapper.compile_cache_entry = patched
    return dict(libneuronxla_version=__version__, compiler_wrapper_sha256=digest,
                compatibility="supply_missing_optional_wrapped_output_path",
                installed_files_changed=False)


@contextmanager
def _uncached_device_state_transfer(runtime):
    """Mutable optimizer counters must not share cached scalar buffers.

    Restored equal CPU scalars otherwise become a single XLA device input,
    unlike the independently updated counters in an uninterrupted optimizer.
    The installed syncfree kernel mutates these inputs. Restore the previous
    cache setting even if a transfer or materialization fails.
    """
    api = runtime._XLAC
    previous = api._get_xla_enable_device_data_cache()
    try:
        api._set_xla_enable_device_data_cache(False)
        yield
        runtime.sync(wait=True)
    finally:
        api._set_xla_enable_device_data_cache(previous)


def restore_syncfree_device_state(optimizer) -> None:
    """Restore all syncfree AdamW state tensors to their shard's device.

    PyTorch AdamW's ordinary load_state_dict leaves scalar step on CPU when
    capturable=False; syncfree AdamW requires it on XLA along with moments.
    Call after ZeRO.load_state_dict and before the next step. This copies values
    without changing precision, optimizer configuration or saved state.
    """
    import torch

    from contextlib import nullcontext

    base = optimizer.base_optimizer
    context = nullcontext()
    if any(parameter.device.type == 'xla' for parameter in base.state):
        import torch_xla
        context = _uncached_device_state_transfer(torch_xla)
    with context:
        for parameter, state in base.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=parameter.device)
