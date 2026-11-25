import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

from tokenizers import Tokenizer

import config
from generate import (
    GenerationResult,
    build_generation_prefix,
    generate_with_model,
    main,
)
from model import GPT, GPTConfig

TOKENIZER_PATH = config.TOKENIZER_DIR / "tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER_PATH.exists():
        pytest.skip("tokenizer not trained")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


class ScriptedModel:
    """A tiny stand-in that forces a known sequence of next-token ids."""

    def __init__(self, tokens, block_size=32):
        self.config = type("Config", (), {"block_size": block_size})()
        self.tokens = list(tokens)
        self.calls = 0

    def __call__(self, idx):
        position = min(self.calls, len(self.tokens) - 1)
        token = self.tokens[position]
        self.calls += 1
        logits = torch.full((1, idx.shape[1], 32_768), -1e4, dtype=torch.float32)
        logits[0, -1, token] = 0.0
        return logits, None


def test_continue_prefix_does_not_append_eot(tok):
    prompt = "The key innovation in machine learning is"
    prompt_ids, prefix_ids, _, truncated = build_generation_prefix(
        tok, prompt, mode="continue", block_size=64
    )

    eot = tok.token_to_id("<|endoftext|>")
    assert prompt_ids == tok.encode(prompt).ids
    assert prefix_ids == prompt_ids
    assert eot not in prefix_ids
    assert not truncated


def test_document_prefix_is_exactly_one_eot(tok):
    eot = tok.token_to_id("<|endoftext|>")
    prompt_ids, prefix_ids, _, truncated = build_generation_prefix(
        tok, "", mode="document", block_size=64
    )

    assert prompt_ids == []
    assert prefix_ids == [eot]
    assert prefix_ids.count(eot) == 1
    assert not truncated


def test_generated_eot_terminates_document_generation(tok):
    generated = [123, 456, 0, 789]
    result = generate_with_model(
        ScriptedModel(generated),
        tok,
        "",
        mode="document",
        max_new_tokens=10,
        decoding="greedy",
        seed=0,
    )

    assert isinstance(result, GenerationResult)
    assert result.generated_token_ids == (123, 456, 0)
    assert result.continuation_token_ids == (123, 456)
    assert result.stop_reason == "eos"
    assert result.eos_position == 2
    assert 789 not in result.generated_token_ids


def test_prompt_and_continuation_are_returned_separately(tok):
    prompt = "We propose a method for"
    generated = tok.encode(" learning compact models.").ids
    result = generate_with_model(
        ScriptedModel(generated),
        tok,
        prompt,
        mode="continue",
        max_new_tokens=len(generated),
        decoding="greedy",
        seed=0,
    )

    assert result.prompt_text == prompt
    assert result.continuation_text.startswith(" learning")
    assert result.text == result.prompt_text + result.continuation_text
    assert result.text == prompt + " learning compact models."
    assert result.eot_id not in result.generated_token_ids


def test_special_tokens_cannot_fuse_prompt_and_document(tok):
    title = "DenseNet: An efficient and robust kernel"
    generated = tok.encode(title).ids
    result = generate_with_model(
        ScriptedModel(generated),
        tok,
        "",
        mode="document",
        max_new_tokens=len(generated),
        decoding="greedy",
        seed=0,
    )

    assert result.prompt_text == ""
    assert result.text == title
    assert not result.text.startswith("The key innovation")
    assert "<|endoftext|>" not in result.text


def test_empty_continue_prompt_is_rejected_clearly(tok):
    with pytest.raises(ValueError, match="empty prompt"):
        build_generation_prefix(tok, "", mode="continue", block_size=16)


def test_long_prompt_is_truncated_safely_from_the_left(tok):
    prompt = "start " + "context " * 40 + "end"
    prompt_ids, prefix_ids, suffix_ids, truncated = build_generation_prefix(
        tok, prompt, mode="continue", block_size=16
    )

    assert truncated is True
    assert len(prefix_ids) == 15
    assert prefix_ids == prompt_ids
    assert prefix_ids[-1] == tok.encode(prompt).ids[-1]
    assert tok.token_to_id("<|endoftext|>") not in prefix_ids
    assert suffix_ids == []


def test_document_mode_rejects_a_nonempty_prompt(tok):
    with pytest.raises(ValueError, match="document mode"):
        build_generation_prefix(tok, "not allowed", mode="document", block_size=16)


def test_prefix_and_suffix_never_contain_two_eots(tok):
    eot = tok.token_to_id("<|endoftext|>")

    for mode, prompt in (("document", ""), ("continue", "A short prompt")):
        _, prefix_ids, suffix_ids, _ = build_generation_prefix(
            tok, prompt, mode=mode, block_size=64
        )
        combined = list(prefix_ids) + list(suffix_ids)
        assert combined.count(eot) <= 1, (mode, combined)
        assert not any(a == eot and b == eot for a, b in zip(combined, combined[1:]))
        assert suffix_ids == []


def test_cli_document_mode_json_starts_with_one_eot(tmp_path, tok, capsys):
    cfg = GPTConfig(
        vocab_size=tok.get_vocab_size(),
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    checkpoint = {
        "config": asdict(cfg),
        "model": GPT(cfg).state_dict(),
        "val_loss": 0.0,
        "step": 0,
    }
    path = tmp_path / "tiny.pt"
    torch.save(checkpoint, path)

    main(
        [
            "--checkpoint",
            str(path),
            "--mode",
            "document",
            "--max-tokens",
            "2",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "document"
    assert payload["prefix_token_ids"] == [tok.token_to_id("<|endoftext|>")]
    assert payload["suffix_token_ids"] == []
    assert payload["eot_id"] == 0

