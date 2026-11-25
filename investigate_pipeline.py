"""Read-only pipeline investigation. Does not modify data, tokenizer, or checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, "src")
from model import GPT, GPTConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
EOT = 0


def show(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


root = Path(".")
tok = Tokenizer.from_file("tokenizer/tokenizer.json")

show("1. TOKENIZER IDENTITY / SPECIAL TOKENS")
print("vocab_size           :", tok.get_vocab_size())
print("eot id / string      :", tok.token_to_id("<|endoftext|>"))
print("pad id / string      :", tok.token_to_id("<|pad|>"))
print("decode([eot])        :", repr(tok.decode([EOT])))
print("decode([pad])        :", repr(tok.decode([1])))
print("decode([eot], keep)  :", repr(tok.decode([EOT], skip_special_tokens=False)))
print("padding              :", repr(tok.padding))
print("truncation           :", repr(tok.truncation))

show("2. TOKENIZER ROUND TRIPS")
samples = [
    "Learning to Reason\n\nWe study how compact models learn.",
    "A Survey of Sparse Attention\n\nTransformers have demonstrated strong results, e.g. 3.14% gains.",
    "  leading and trailing spaces  ",
    "line one\nline two\n\nline four",
    "Curie-Weiss model: $\\alpha^2 + \\beta_1 \\to \\infty$",
]
for s in samples:
    enc = tok.encode(s)
    dec = tok.decode(enc.ids)
    print("\nORIGINAL :", repr(s))
    print("ids[:20] :", enc.ids[:20])
    print("tokens   :", enc.tokens[:12])
    print("DECODED  :", repr(dec))
    print("EXACT    :", dec == s)
    if dec != s:
        for i, (a, b) in enumerate(zip(s, dec)):
            if a != b:
                print("first diff at", i, repr(s[max(0, i - 15):i + 15]), "->", repr(dec[max(0, i - 15):i + 15]))
                break
        else:
            print("length differs:", len(s), len(dec))

show("3. WHITESPACE / PREFIX-SPACE BEHAVIOUR")
for s in ["a b", "a  b", "a\n\nb", " learning", "Learning", "is", " is", "isLearning", "is Learning"]:
    ids = tok.encode(s).ids
    pieces = [tok.decode([i]) for i in ids]
    print(f"{s!r:18} ids={ids} pieces={pieces} decoded={tok.decode(ids)!r}")

show("4. CORPUS FORMATTING (processed parquet)")
train = pd.read_parquet("data/processed/train.parquet")
print("columns:", list(train.columns))
print("rows    :", len(train))
for i in range(3):
    r = train.iloc[i]
    text = r["text"]
    print(f"\n--- record {i} id={r['id']} ---")
    print("repr[:260] :", repr(text[:260]))
    print("len chars  :", len(text))
    print("has '\\n\\n':", "\n\n" in text, "| count:", text.count("\n\n"))
    print("ends with  :", repr(text[-40:]))
    print("leading ws :", repr(text[:5]), "trailing ws:", repr(text[-5:]))
    print("literal EOT string in text:", "<|endoftext|>" in text)

print("\ntitle/abstract separator check over 2000 rows:")
sep = train["text"].head(2000).str.contains("\n\n", regex=False)
print("  rows containing '\\n\\n':", int(sep.sum()), "/", len(sep))
print("  rows with tab:", int(train["text"].head(2000).str.contains("\t", regex=False).sum()))
print("  rows with double space:", int(train["text"].head(2000).str.contains("  ", regex=False).sum()))

show("5. TOKENIZED VALIDATION DOCUMENTS / EOT BOUNDARIES")
val = np.memmap("data/subsets/val.bin", dtype=np.uint16, mode="r")
val_docs = pd.read_parquet("data/subsets/val_docs.parquet")
print("val.bin tokens:", len(val), "docs:", len(val_docs))
print("last token id (should be EOT=0):", int(val[-1]))
print("total EOT count in val.bin:", int((np.asarray(val) == EOT).sum()))

for i in range(3):
    row = val_docs.iloc[i]
    s, e = int(row["start"]), int(row["end"])
    doc_ids = val[s:e - 1].tolist()
    print(f"\n--- doc {i} id={row['id']} offsets=[{s},{e}) len={e-s} ---")
    print("last id of span (EOT expected):", int(val[e - 1]))
    print("first id of next doc         :", int(val[e]) if e < len(val) else "N/A")
    print("decoded doc[:200]            :", repr(tok.decode(doc_ids)[:200]))
    src = train[train["id"] == row["id"]]
    if len(src):
        print("source text[:120]            :", repr(src.iloc[0]["text"][:120]))
        print("decoded == source            :", tok.decode(doc_ids) == src.iloc[0]["text"])

print("\nboundary audit over first 500 docs:")
ok_last = ok_next = ok_inner = 0
for i in range(min(500, len(val_docs))):
    row = val_docs.iloc[i]
    s, e = int(row["start"]), int(row["end"])
    if int(val[e - 1]) == EOT:
        ok_last += 1
    if e >= len(val) or int(val[e]) != EOT:
        ok_next += 1
    if EOT not in np.asarray(val[s:e - 1]):
        ok_inner += 1
print("  docs whose final id is EOT      :", ok_last, "/500")
print("  docs whose next id is not EOT   :", ok_next, "/500  (EOT only at doc end)")
print("  docs with no EOT in body        :", ok_inner, "/500")

show("6. TRAINING WINDOWS CAN CROSS DOCUMENT BOUNDARIES")
# Count how many 512-token windows contain an EOT in positions 1..510.
bs = 512
n_win = (len(val) - 1) // bs
arr = np.asarray(val[: n_win * bs])
w = arr[: n_win * bs].reshape(n_win, bs)
inner_eot = (w[:, 1:bs - 1] == EOT).sum(axis=1)
print("windows:", n_win)
print("windows containing >=1 internal EOT:", int((inner_eot > 0).sum()), f"({(inner_eot > 0).mean():.1%})")
print("mean internal EOTs per window      :", float(inner_eot.mean()))
print("=> Targets are unmasked across document boundaries; one window can contain many documents.")

show("7. CHECKPOINT RESTORATION")
runs = sorted(Path("runs").glob("*/seed*"))
for run in runs:
    p = run / "ckpt_best.pt"
    if not p.exists():
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    missing = model.load_state_dict(ck["model"], strict=False)
    print(f"{run.as_posix():28} params={model.num_parameters()/1e6:.3f}M step={ck.get('step')} "
          f"val_loss={ck.get('val_loss')} cfg={ck['config']} "
          f"keys={sorted(ck.keys())} load_missing={list(missing.missing_keys)} "
          f"load_unexpected={list(missing.unexpected_keys)}")

show("8. GENERATION TOKEN-LEVEL INSTRUMENTATION (one checkpoint)")
ck_path = Path("runs/tail_aware/seed42/ckpt_best.pt")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
cfg = GPTConfig(**ck["config"])
model = GPT(cfg)
model.load_state_dict(ck["model"])
model.eval()

PROMPT = "The key innovation in machine learning is"
MAX_NEW = 40


def run_gen(prompt, mode, max_new=MAX_NEW, temperature=0.7, seed=0):
    torch.manual_seed(seed)
    pids = tok.encode(prompt).ids
    if mode == "eot2":
        ids = pids[: cfg.block_size - 2] + [EOT, EOT]
    elif mode == "eot1":
        ids = pids[: cfg.block_size - 1] + [EOT]
    elif mode == "newline":
        ids = (pids + tok.encode("\n\n").ids)[: cfg.block_size]
    elif mode == "none":
        ids = list(pids)
    else:
        raise ValueError(mode)

    prefix_ids = list(ids)
    generated, probs = [], []
    with torch.no_grad():
        for _ in range(max_new):
            if len(ids) > cfg.block_size:
                ids = ids[-cfg.block_size:]
            logits, _ = model(torch.tensor([ids], dtype=torch.long))
            nxt = logits[0, -1] / temperature
            p = torch.softmax(nxt, dim=-1)
            t = int(torch.multinomial(p, 1).item())
            generated.append(t)
            probs.append(float(p[t]))
            if t == EOT:
                break
            ids.append(t)
    cont = [t for t in generated if t != EOT]
    return {
        "mode": mode,
        "prompt_ids": pids,
        "prefix_ids": prefix_ids,
        "generated_ids": generated,
        "cont_ids": cont,
        "eos_in_generated": generated.count(EOT),
        "stopped_at_eos": bool(generated and generated[-1] == EOT),
        "prompt_only_decode": tok.decode(pids),
        "prefix_only_decode": tok.decode(prefix_ids),
        "prefix_decode_noskip": tok.decode(prefix_ids, skip_special_tokens=False),
        "cont_decode": tok.decode(cont),
        "full_decode_skip": tok.decode(prefix_ids + generated, skip_special_tokens=True),
        "full_decode_keep": tok.decode(prefix_ids + generated, skip_special_tokens=False),
        "first_20_pieces": [(t, tok.decode([t])) for t in generated[:20]],
        "first_20_probs": [round(x, 4) for x in probs[:20]],
    }


for mode in ("eot2", "eot1", "newline", "none"):
    r = run_gen(PROMPT, mode)
    print(f"\n--- prompt_mode={mode} ---")
    print("prompt ids         :", r["prompt_ids"])
    print("prefix ids         :", r["prefix_ids"][-8:], "(last 8)")
    print("generated ids (40) :", r["generated_ids"][:20], "...")
    print("EOT in generated   :", r["eos_in_generated"], "| stopped at EOS:", r["stopped_at_eos"])
    print("first 20 (id,piece):", r["first_20_pieces"])
    print("first 20 probs     :", r["first_20_probs"])
    print("prompt decode      :", repr(r["prompt_only_decode"][-40:]))
    print("prefix decode      :", repr(r["prefix_only_decode"][-60:]))
    print("prefix decode keep :", repr(r["prefix_decode_noskip"][-60:]))
    print("continuation decode:", repr(r["cont_decode"][:160]))
    print("FULL decode (skip) :", repr(r["full_decode_skip"][:200]))
    print("FULL decode (keep) :", repr(r["full_decode_keep"][:200]))

show("9. IS THE FUSION 'is' + 'Title' CAUSED BY SKIPPED EOT?")
r = run_gen(PROMPT, "eot2", max_new=20)
print("prefix last ids:", r["prefix_ids"][-4:])
print("decode(prefix)          :", repr(r["prefix_only_decode"]))
print("decode(prefix, keep EOT):", repr(r["prefix_decode_noskip"]))
print("decode(generated only)  :", repr(r["cont_decode"]))
print("decode(prefix+gen, skip):", repr(r["full_decode_skip"]))
print("decode(prefix+gen, keep):", repr(r["full_decode_keep"]))
print("\nNOTE: if decode(prefix,keep) ends with literal '<|endoftext|>' and the")
print("skip version has no separator, the missing space is a decode artefact")
print("of the artificial EOT tokens, not the tokenizer dropping spaces.")
