"""Random architecture initialization and bounded CPU qualification reference."""
import argparse,json
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from train.neuron_backend import NeuronCausalBackend
from train.pause_pretraining import objective,pack,source_digest
from train.runtime_utils import cast_parameters,tensor_digest,sampled_fingerprint
from train.artifacts import atomic_json,sha256_file

def model_from_config(path, seed=2200):
    torch.manual_seed(seed)
    config = Qwen3Config.from_json_file(path)
    if (config.hidden_size, config.intermediate_size, config.num_hidden_layers,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        config.vocab_size, config.tie_word_embeddings) != (1024, 3072, 28, 16, 8, 128, 151936, True):
        raise ValueError('Exact official Qwen3-0.6B architecture required')
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).eval()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def model_digest(model):
    return {k: tensor_digest(t) for k, t in model.state_dict().items()}


def reference(a, rows):
    model = model_from_config(a.config)
    cast_parameters(model)
    fingerprint = sampled_fingerprint(model)
    # Canonical FP32 compute on exactly the BF16-quantized initialization.
    for param in model.parameters():
        param.data = param.data.float()
    backend = NeuronCausalBackend(model, head_chunk=128, checkpoint_trunk=False)
    ids = torch.tensor([r['ids'] for r in rows[:2]])
    values = {}
    with torch.no_grad():
        for active in (False, True):
            batch = pack(ids, pauses=active)
            result = objective(backend, batch)
            values[str(active)] = {'loss': float(result.loss),
                'per_example': result.per_example['source'].tolist()}
    receipt = {'initial_fingerprint': fingerprint, 'references': values,
        'head_chunk': 128,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'config_sha256': sha256_file(a.config), 'data_sha256': sha256_file(a.data),
        'ordered_source_digest': source_digest(rows)}
    atomic_json(a.output/a.reference_name, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('reference',),default='reference')
    p.add_argument('--config',type=Path,default=Path('configs/qwen3_0.6b_architecture.json'))
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-name',default='cpu-reference-whole.json')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.data.read_text())
    if len(rows)!=768 or any(len(r['ids'])!=513 for r in rows):
        raise ValueError('Exactly 768 fixed qualification windows required')
    reference(a,rows)

if __name__=='__main__':
    main()
