"""Train the byte-level BPE tokenizer on the train split only, then count tokens.

Outputs
  tokenizer/tokenizer.json              the fixed tokenizer (committed)
  data/selection/token_counts.parquet   id, split, n_tokens (EOT not included)
  reports/tokenizer_report.json         vocab-size comparison and fertility
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

import config
from topics import assign_groups, category_groups

COMPARE_SIZES = (8_192, 16_384, 32_768)
COMPARE_SAMPLE = 100_000


def batches(texts, size=1_000):
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


def train_bpe(texts, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[config.EOT_TOKEN, config.PAD_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(batches(texts), trainer=trainer, length=len(texts))
    return tok


def count_tokens(tok, texts):
    out = np.empty(len(texts), dtype=np.int32)
    for i, chunk in enumerate(batches(texts, 10_000)):
        enc = tok.encode_batch(chunk)
        out[i * 10_000 : i * 10_000 + len(chunk)] = [len(e.ids) for e in enc]
    return out


def fertility(n_tokens, texts):
    """Tokens per whitespace-separated word (lower = more efficient)."""
    words = pd.Series(texts).str.split().str.len().to_numpy()
    return float(n_tokens.sum() / words.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compare", action="store_true", help="skip the vocab-size comparison")
    args = ap.parse_args()

    config.ensure_dirs()
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet", columns=["id", "text", "primary_category"])
    val = pd.read_parquet(config.PROCESSED_DIR / "val.parquet", columns=["id", "text", "primary_category"])
    test = pd.read_parquet(config.PROCESSED_DIR / "test.parquet", columns=["id", "text"])
    report = {"train_docs": len(train), "special_tokens": [config.EOT_TOKEN, config.PAD_TOKEN]}

    # 1. Vocab-size comparison on a random train sample, measured on val.
    if not args.skip_compare:
        sample = train["text"].sample(COMPARE_SAMPLE, random_state=config.SEED).tolist()
        val_texts = val["text"].tolist()
        rows = []
        for v in COMPARE_SIZES:
            t0 = time.time()
            tok = train_bpe(sample, v)
            fert = fertility(count_tokens(tok, val_texts), val_texts)
            rows.append({"vocab_size": v, "val_tokens_per_word": round(fert, 4),
                         "embedding_params_d384": v * 384, "train_seconds": round(time.time() - t0, 1)})
            print(rows[-1])
        report["vocab_comparison"] = {"train_sample_docs": COMPARE_SAMPLE, "rows": rows}

    # 2. The fixed tokenizer: full train split, nothing from val / test.
    t0 = time.time()
    tok = train_bpe(train["text"].tolist(), config.VOCAB_SIZE)
    report["final_train_seconds"] = round(time.time() - t0, 1)
    path = config.TOKENIZER_DIR / "tokenizer.json"
    tok.save(str(path))
    report["vocab_size"] = tok.get_vocab_size()
    report["eot_id"] = tok.token_to_id(config.EOT_TOKEN)
    report["pad_id"] = tok.token_to_id(config.PAD_TOKEN)
    print(f"Saved {path} (vocab {report['vocab_size']}, {report['final_train_seconds']}s)")

    # 3. Token counts for every document in every split.
    parts = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        n = count_tokens(tok, df["text"].tolist())
        parts.append(pd.DataFrame({"id": df["id"], "split": name, "n_tokens": n}))
        report[f"{name}_tokens"] = int(n.sum())
        report[f"{name}_tokens_per_word"] = round(fertility(n, df["text"].tolist()), 4)
    counts = pd.concat(parts, ignore_index=True)
    counts.to_parquet(config.SELECTION_DIR / "token_counts.parquet", index=False)

    # 4. Does the tokenizer handle tail topics as efficiently as head topics? (val)
    cat_to_group = category_groups(train["primary_category"])
    vc = counts[counts["split"] == "val"].reset_index(drop=True)
    val_groups = assign_groups(val["primary_category"], cat_to_group).to_numpy()
    report["val_tokens_per_word_by_group"] = {
        g: round(fertility(vc["n_tokens"].to_numpy()[val_groups == g], val["text"][val_groups == g].tolist()), 4)
        for g in ("head", "mid", "tail")
    }

    sample_text = val["text"].iloc[0][:300]
    report["example"] = {"text": sample_text, "tokens": tok.encode(sample_text).tokens[:60]}
    (config.REPORTS_DIR / "tokenizer_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "example"}, indent=2))


if __name__ == "__main__":
    main()
