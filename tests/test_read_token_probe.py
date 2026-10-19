import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.mark.slow
def test_cached_left_padded_read_state_matches_full_forward():
    from scripts.read_token_probe import ReadToken, read_states
    name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                                  local_files_only=True).eval()
    except OSError:
        pytest.skip("model not cached")
    d = model.config.hidden_size
    rt = ReadToken(3, d, torch.randn(3, d) * 0.02)
    seqs = [tok("Problem: 2+2. Step: 4.")["input_ids"], tok("Hi")["input_ids"]]
    got = read_states(model, rt, seqs, tok.eos_token_id, torch.device("cpu"), grad=True)
    got.sum().backward()
    assert rt.soft.grad is not None and rt.soft.grad.abs().sum() > 0
    for r, s in enumerate(seqs):
        e = model.get_input_embeddings()(torch.tensor([s]))
        full = torch.cat([e, rt.soft.detach()[None]], 1)
        ref = model.model(inputs_embeds=full).last_hidden_state[0, -1]
        assert torch.allclose(got[r].detach(), ref, atol=2e-3), (r, (got[r] - ref).abs().max())
