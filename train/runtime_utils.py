"""Unchanged precision/fingerprint/digest primitives from the qualified predecessor."""
import hashlib
import torch
from train.artifacts import canonical_hash
STATE_FIELDS=("exp_avg","exp_avg_sq","max_exp_avg_sq","step")

def cast_parameters(model):
    """CPU only; preserve tied Parameter objects and byte-exact FP32 buffers."""
    parameters=list(model.parameters());identities=[id(p) for p in parameters]
    if not parameters or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in parameters):
        raise ValueError('Cast released CPU FP32 parameters before any device transfer')
    buffers={name:t.detach().clone() for name,t in model.named_buffers()}
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in buffers.values()):
        raise ValueError('Original floating buffers must remain FP32')
    with torch.no_grad():
        for p in parameters: p.data=p.data.to(torch.bfloat16)
    if [id(p) for p in model.parameters()]!=identities:
        raise RuntimeError('Parameter identity changed during cast')
    for name,t in model.named_buffers():
        old=buffers[name]
        if t.dtype!=old.dtype or not torch.equal(t.contiguous().view(torch.uint8),old.contiguous().view(torch.uint8)):
            raise RuntimeError('CPU parameter cast altered a buffer')
    return dict(parameters=len(parameters),parameter_dtype='torch.bfloat16',
        buffers={name:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=canonical_hash(t.reshape(-1).tolist()))
                 for name,t in buffers.items()},parameter_objects_preserved=True)


def tensor_digest(value):
    """Hash actual CPU bytes, including BF16 without converting its values."""
    import torch
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
    h.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def optimizer_digest(state):
    """All masters, moments and Adam counters, with no sampled-only claim."""
    masters=state['sharded_master_weights'];moments=state['base_state']
    if set(masters)!=set(moments) or any(set(value)!=set(STATE_FIELDS) for value in moments.values()):
        raise ValueError('Installed syncfree AdamW tensor inventory differs')
    return dict(param_groups=canonical_hash(state['param_groups']),
        tensors={str(i):dict(master=tensor_digest(master),
            **{key:tensor_digest(moments[i][key]) for key in STATE_FIELDS})
            for i,master in masters.items()})


def sampled_fingerprint(model) -> dict:
    """A cheap integrity signature, NOT a full cryptographic weight-file hash."""
    h=hashlib.sha256();count=0
    for name,tensor in model.named_parameters():
        flat=tensor.detach().reshape(-1)
        # Floating linspace rounds the final index up for 300M-element
        # embeddings. Compute evenly spaced integer indices exactly on host.
        n=flat.numel(); k=min(32,n)
        idx=torch.tensor([i*(n-1)//max(1,k-1) for i in range(k)],
                         device=flat.device,dtype=torch.long)
        h.update(name.encode());h.update(flat[idx].float().cpu().numpy().tobytes());count+=tensor.numel()
    return {'sampled_parameter_sha256':h.hexdigest(),'parameters':count,
            'qualification':'Compare at initialization and after CPU/device transfer; also record full source-file SHA256s.'}

