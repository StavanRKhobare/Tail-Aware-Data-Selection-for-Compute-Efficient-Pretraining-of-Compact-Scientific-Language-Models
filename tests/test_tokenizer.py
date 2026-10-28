import pytest
from tokenizers import Tokenizer

import config

PATH = config.TOKENIZER_DIR / "tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    if not PATH.exists():
        pytest.skip("tokenizer not trained yet (run src/train_tokenizer.py)")
    return Tokenizer.from_file(str(PATH))


def test_vocab_and_special_tokens(tok):
    assert tok.get_vocab_size() == config.VOCAB_SIZE
    assert tok.token_to_id(config.EOT_TOKEN) == 0
    assert tok.token_to_id(config.PAD_TOKEN) == 1


@pytest.mark.parametrize("text", [
    "Attention Is All You Need\n\nWe propose the Transformer.",
    "We prove an $O(\\sqrt{T}\\log T)$ regret bound for $\\epsilon$-greedy exploration.",
    "Résumé: naïve Bayes über alles — 100% of 3.14 cases.",
])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_common_scientific_words_are_single_tokens(tok):
    for word in [" learning", " neural", " language", " reinforcement", " transformer"]:
        assert len(tok.encode(word).ids) == 1, word
