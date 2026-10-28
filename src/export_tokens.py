"""Tokenize each subset (and val / test) into flat uint16 files for Part 3.

Format (nanoGPT-style): every document's token ids followed by one
<|endoftext|> id, concatenated into a single uint16 array.

Outputs (data/subsets/, git-ignored, rebuild with this script)
  <name>/train.bin        subset documents in a seeded shuffled order
  val.bin, test.bin       full val / test splits in id order
  val_docs.parquet, test_docs.parquet
                          id, primary_category, group, cluster, start, end
                          (token offsets, for topic-wise evaluation in Part 4)
  meta.json               vocab size, special ids, token counts
"""
import gzip
import json

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

import config
from select_subsets import STRATEGIES


def encode(tok, texts, eot):
    ids, lengths = [], []
    for i in range(0, len(texts), 10_000):
        for e in tok.encode_batch(texts[i : i + 10_000]):
            ids.append(np.asarray(e.ids + [eot], dtype=np.uint16))
            lengths.append(len(e.ids) + 1)
    return np.concatenate(ids), np.asarray(lengths, dtype=np.int64)


def main():
    config.ensure_dirs()
    tok = Tokenizer.from_file(str(config.TOKENIZER_DIR / "tokenizer.json"))
    eot = tok.token_to_id(config.EOT_TOKEN)
    assert tok.get_vocab_size() <= np.iinfo(np.uint16).max + 1
    meta = {"vocab_size": tok.get_vocab_size(), "eot_id": eot, "pad_id": tok.token_to_id(config.PAD_TOKEN),
            "dtype": "uint16", "token_budget": config.TOKEN_BUDGET, "files": {}}

    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet", columns=["id", "text"]).set_index("id")
    for name in STRATEGIES:
        manifest = config.MANIFEST_DIR / f"{name}.txt.gz"
        ids = gzip.open(manifest, "rt").read().split()
        order = np.random.default_rng(config.SEED).permutation(len(ids))
        texts = train.loc[np.asarray(ids)[order], "text"].tolist()
        arr, _ = encode(tok, texts, eot)
        out = config.SUBSETS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        arr.tofile(out / "train.bin")
        meta["files"][f"{name}/train.bin"] = {"docs": len(ids), "tokens": int(len(arr))}
        print(name, meta["files"][f"{name}/train.bin"])

    topics = pd.read_parquet(config.SELECTION_DIR / "topics.parquet")
    for split in ("val", "test"):
        df = pd.read_parquet(config.PROCESSED_DIR / f"{split}.parquet", columns=["id", "text"]).sort_values("id")
        arr, lengths = encode(tok, df["text"].tolist(), eot)
        arr.tofile(config.SUBSETS_DIR / f"{split}.bin")
        end = np.cumsum(lengths)
        docs = df[["id"]].assign(start=end - lengths, end=end).merge(
            topics[topics["split"] == split].drop(columns="split"), on="id", how="left")
        docs.to_parquet(config.SUBSETS_DIR / f"{split}_docs.parquet", index=False)
        meta["files"][f"{split}.bin"] = {"docs": len(df), "tokens": int(len(arr))}
        print(split, meta["files"][f"{split}.bin"])

    (config.SUBSETS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
