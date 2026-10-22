import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.neuron_backend import NeuronCausalBackend
from train.pause_pretraining import (PAUSE_ID, arithmetic_program, enabled,
    estimated_flops, learning_rate, objective, pack, source_digest, task_window)


def tiny_model():
    torch.manual_seed(7)
    c = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, attention_dropout=0., tie_word_embeddings=True)
    c._attn_implementation = 'eager'
    return Qwen3ForCausalLM(c).eval()


def test_boundary_shift_no_drop_and_no_trailing_compute():
    real = torch.arange(513)[None]
    b = pack(real, pauses=True)
    assert b.input_ids.shape == (1, 577) and b.pause_count == 64
    assert torch.equal(b.input_ids[:, b.real_positions], real)
    assert b.target_positions[0, 127] == 144  # target real token 128
    assert b.input_ids[0, 143] == PAUSE_ID  # predictor, last pause
    assert b.input_ids[0, 144] == 128
    assert torch.equal(b.labels, real[:, 1:])
    assert pack(real[:, :128], pauses=True).pause_count == 0
    assert pack(real[:, :129], pauses=True).pause_count == 16
    assert pack(real[:, :512], pauses=True).pause_count == 48
    assert torch.equal(pack(real, pauses=False).input_ids, real)


def test_ce_equals_independent_hf_reference_and_all_real_targets():
    model = tiny_model()
    real = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [7, 4, 8, 2, 1, 9, 3]])
    backend = NeuronCausalBackend(model, checkpoint_trunk=False, head_chunk=3)
    raw = objective(backend, pack(real, pauses=False)).loss
    hf = model(input_ids=real, labels=real).loss
    torch.testing.assert_close(raw, hf)
    b = pack(real, pauses=True, interval=3, count=2, pause_id=40)
    got = objective(backend, b)
    # Separate full-HF forward, explicitly indexing each predecessor.
    logits = model(input_ids=b.input_ids).logits
    selected = logits[:, b.target_positions[0]-1]
    expected = torch.nn.functional.cross_entropy(selected.flatten(0, 1), real[:, 1:].flatten())
    torch.testing.assert_close(got.loss, expected)
    assert got.counts['source'].tolist() == [6., 6.]


def test_causal_prefix_and_gradient_through_pause_inputs():
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=True, head_chunk=4)
    real = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    b = pack(real, pauses=True, interval=3, count=2, pause_id=40)
    changed = real.clone(); changed[:, 4:] = 9
    c = pack(changed, pauses=True, interval=3, count=2, pause_id=40)
    with torch.no_grad():
        hb, hc = backend.hidden(b.input_ids), backend.hidden(c.input_ids)
        torch.testing.assert_close(hb[:, :6], hc[:, :6], rtol=0, atol=0)
    # Isolate the INPUT embedding gradient from a tied head's output gradient.
    embedded = []
    handle = model.model.embed_tokens.register_forward_hook(lambda m, a, o: (o.retain_grad(), embedded.append(o)) and None)
    objective(backend, b).loss.backward()
    handle.remove()
    assert embedded[0].grad[:, 3:5].abs().sum() > 0
    assert torch.isfinite(embedded[0].grad).all()


def test_late_switch_preserves_optimizer_and_initial_identity():
    early, vanilla = tiny_model(), tiny_model()
    assert all(torch.equal(a, b) for a, b in zip(early.parameters(), vanilla.parameters()))
    assert [enabled('late', s, 3) for s in range(6)] == [False]*3+[True]*3
    assert all(enabled('early', s, 3) for s in range(6))
    assert not any(enabled('vanilla', s, 3) for s in range(6))
    opt = torch.optim.AdamW(early.parameters(), lr=.001)
    backend = NeuronCausalBackend(early, checkpoint_trunk=False)
    real = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    for step in range(2):
        opt.zero_grad()
        objective(backend, pack(real, pauses=enabled('late', step, 1),
            interval=3, count=2, pause_id=40)).loss.backward()
        opt.step()
    assert all(float(s['step']) == 2 for s in opt.state.values())


def test_task_interpreter_namespaces_and_answer_alignment():
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, **kw):
            return [ord(c) for c in text]
    tok = Tokenizer()
    p = arithmetic_program(3, 'train', 4, 4)
    x = p['initial']
    for a, b in p['operations']:
        x = (a*x+b) % 97
    assert p['answer'] == str(x)
    assert p == arithmetic_program(3, 'train', 4, 4)
    assert p['key'] != arithmetic_program(3, 'confirm', 4, 4)['key']
    ids, meta = task_window(tok, p, [65]*600, width=600, boundary=256)
    assert meta['answer_start'] == 256
    assert ids[256:256+len(p['answer'])] == tok.encode(p['answer'])
    wrong = dict(p, answer='1' if p['answer'] != '1' else '2')
    ids2, _ = task_window(tok, wrong, [65]*600, width=600, boundary=256)
    assert ids[:256] == ids2[:256]


def test_budget_and_schedule():
    c = tiny_model().config
    a = estimated_flops(c, real_inputs=512, expanded_inputs=512, targets=512)
    b = estimated_flops(c, real_inputs=512, expanded_inputs=576, targets=512)
    assert b['matmul_flops'] > a['matmul_flops']
    assert learning_rate(0, total=10, warmup=2) == .00015
    assert learning_rate(9, total=10, warmup=2) == pytest.approx(.00003)
    assert source_digest([{'ids': [1, 2]}]) != source_digest([{'ids': [2, 1]}])


def test_full_cpu_resume_across_switch():
    import copy
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=False)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    real = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    def update(step):
        opt.zero_grad(set_to_none=True)
        objective(backend, pack(real, pauses=enabled('late', step, 1),
            interval=3, count=2, pause_id=40)).loss.backward()
        opt.step()
    update(0)
    state = copy.deepcopy((model.state_dict(), opt.state_dict()))
    update(1)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(state[0]); opt.load_state_dict(state[1])
    update(1)
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())


def test_cached_greedy_uses_boundary_pauses_and_matches_full_prefix():
    from train.pause_pretraining_evaluate import greedy_answer, score_windows
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=False)
    prefix = torch.tensor([[1, 2, 3]])
    got = greedy_answer(backend, prefix, pauses=True, stop_id=0, max_new_tokens=1,
        interval=3, count=2, pause_id=40)
    extended = torch.tensor([[1, 2, 3, 40, 40]])
    with torch.no_grad():
        logits = model(input_ids=extended).logits[:, -1]
    logits[:, 40] = -torch.inf
    assert torch.equal(got[:, 0], logits.argmax(-1))
    real = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    scores = score_windows(backend, real, pauses=False, answer_starts=[3], answer_lengths=[2])
    torch.testing.assert_close(scores['full_NLL'].mean(), model(input_ids=real, labels=real).loss)


def test_multitoken_cached_generation_crosses_new_pause_boundary():
    from train.pause_pretraining_evaluate import greedy_answer
    model = tiny_model()
    backend = NeuronCausalBackend(model, checkpoint_trunk=False)
    prefix = torch.tensor([[1, 2]])
    got = greedy_answer(backend, prefix, pauses=True, stop_id=0, max_new_tokens=4,
        interval=3, count=2, pause_id=40)
    real = prefix.clone()
    expected, alive = [], True
    for index in range(4):
        b = pack(real, pauses=True, interval=3, count=2, pause_id=40)
        x = b.input_ids
        if real.shape[1] % 3 == 0:
            x = torch.cat((x, torch.full((1, 2), 40)), 1)
        with torch.no_grad():
            logits = model(input_ids=x).logits[:, -1]
        logits[:, 40] = -torch.inf
        token = int(logits.argmax(-1)) if alive else 0
        expected.append(token); alive = alive and token != 0
        real = torch.cat((real, torch.tensor([[token]])), 1)
    assert got.tolist() == [expected]


def test_whole_instance_source_order_has_no_duplicate_or_missing_targets():
    from train.pause_pretraining_distributed import source_indices
    consumed = [source_indices(step, rank, micro)
        for step in range(6) for micro in range(2) for rank in range(64)]
    assert consumed == list(range(768))


@pytest.mark.parametrize('value', [torch.tensor([[1, PAUSE_ID]]), torch.tensor([[1]]), torch.tensor([[1., 2.]])])
def test_reject_invalid_sources(value):
    with pytest.raises(ValueError):
        pack(value, pauses=True)
