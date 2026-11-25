"""Read-only token-level validation loss by head/mid/tail group.

Uses the val.bin windows and the document offsets in val_docs.parquet. Each
target token is attributed to the document that contains it, using that
document's group label. This is a diagnostic approximation: windows can span
documents and attention is not reset at boundaries, exactly as in training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from model import GPT, GPTConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(".")
SUBSETS = ["random", "quality", "diversity", "tail_aware", "tail_only"]
SEEDS = [42, 43]
GROUPS = ["head", "mid", "tail"]
BLOCK = 512
BATCH = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val = np.memmap(ROOT / "data" / "subsets" / "val.bin", dtype=np.uint16, mode="r")
docs = pd.read_parquet(ROOT / "data" / "subsets" / "val_docs.parquet").sort_values("start").reset_index(drop=True)
ends = docs["end"].to_numpy(dtype=np.int64)
group_to_id = {g: i for i, g in enumerate(GROUPS)}
group_ids = docs["group"].map(group_to_id).to_numpy(dtype=np.int64)
n_windows = (len(val) - 1) // BLOCK
starts_all = np.arange(n_windows, dtype=np.int64) * BLOCK

print(f"device={device} val_tokens={len(val):,} docs={len(docs):,} windows={n_windows:,}")
print(f"{'model':11} {'seed':4} {'overall':>8} " + " ".join(f"{g:>8}" for g in GROUPS) + "   n_head/mid/tail")

results = []
for subset in SUBSETS:
    for seed in SEEDS:
        ck = torch.load(ROOT / "runs" / subset / f"seed{seed}" / "ckpt_best.pt",
                        map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        sums = np.zeros(len(GROUPS) + 1, dtype=np.float64)  # last = overall
        counts = np.zeros(len(GROUPS) + 1, dtype=np.int64)
        amp = torch.bfloat16 if device.type == "cuda" else None
        with torch.no_grad():
            for b0 in range(0, n_windows, BATCH):
                idx = starts_all[b0:b0 + BATCH]
                x_np = np.stack([val[s:s + BLOCK] for s in idx]).astype(np.int64)
                y_np = np.stack([val[s + 1:s + 1 + BLOCK] for s in idx]).astype(np.int64)
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)
                if amp is not None:
                    with torch.autocast(device_type=device.type, dtype=amp):
                        logits, _ = model(x)
                else:
                    logits, _ = model(x)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    reduction="none",
                ).reshape(y.shape)
                pos = (idx[:, None] + 1 + np.arange(BLOCK)[None, :]).ravel()
                doc_idx = np.searchsorted(ends, pos, side="right")
                g = group_ids[np.clip(doc_idx, 0, len(group_ids) - 1)]
                w = losses.reshape(-1).double().cpu().numpy()
                sums[:len(GROUPS)] += np.bincount(g, weights=w, minlength=len(GROUPS))
                counts[:len(GROUPS)] += np.bincount(g, minlength=len(GROUPS))
                sums[-1] += w.sum()
                counts[-1] += w.size
        per_group = sums[:len(GROUPS)] / np.maximum(counts[:len(GROUPS)], 1)
        overall = sums[-1] / counts[-1]
        row = {
            "model": subset, "seed": seed, "overall": float(overall),
            **{g: float(per_group[i]) for i, g in enumerate(GROUPS)},
            "tokens": int(counts[-1]),
            "group_token_counts": {g: int(counts[i]) for i, g in enumerate(GROUPS)},
        }
        results.append(row)
        print(f"{subset:11} {seed:4d} {overall:8.4f} " +
              " ".join(f"{per_group[i]:8.4f}" for i in range(len(GROUPS))) +
              f"   {counts[0]:,}/{counts[1]:,}/{counts[2]:,}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

(ROOT / "reports" / "val_loss_by_group.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print("\nwrote reports/val_loss_by_group.json")
