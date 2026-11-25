import pytest

torch = pytest.importorskip("torch")

from model import GPT, GPTConfig


def tiny_config():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_forward_shapes_and_loss():
    model = GPT(tiny_config())
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    x = torch.randint(0, 64, (2, 16))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 64
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_default_config_is_about_17m_params_and_tied():
    model = GPT(GPTConfig())
    assert 17_000_000 < model.num_parameters() < 17_300_000
    assert model.lm_head.weight is model.tok_emb.weight
