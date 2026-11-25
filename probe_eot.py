"""Read-only probe: does each model assign high probability to EOT at document ends?"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from data import TokenWindows  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(".")
EOT = 0
SUBSETS = ["random", "quality", "diversity", "tail_aware", "tail_only"]
N_WINDOWS = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

windows = TokenWindows(ROOT / "data" / "subsets" / "val.bin", 512)
indices = np.unique(np.linspace(0, windows.n_windows - 1, N_WINDOWS).round().astype(np.int64))
x, y = windows.get_batch(indices)
x, y = x.to(device), y.to(device)

print(f"device={device} windows={len(indices)} positions={x.numel():,} EOT targets={int((y==0).sum()):,}")
print(f"{'model':11} {'p(EOT|target=EOT)':>18} {'p(EOT|target!=EOT)':>19} "
      f"{'EOT recall':>11} {'EOT FPR':>8} {'loss@EOT':>10} {'loss@other':>11}")

for subset in SUBSETS:
    ck = torch.load(ROOT / "runs" / subset / "seed42" / "ckpt_best.pt",
                    map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        p_eot = log_probs[:, :, EOT].exp()
        is_eot = y == EOT
        p_at_eot = p_eot[is_eot]
        p_at_other = p_eot[~is_eot]
        pred = logits.argmax(dim=-1)
        recall = (pred[is_eot] == EOT).float().mean()
        fpr = (pred[~is_eot] == EOT).float().mean()
        loss_eot = -log_probs[:, :, EOT][is_eot].mean()
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).reshape(y.shape)
        loss_other = ce[~is_eot].mean()
    print(f"{subset:11} {p_at_eot.mean().item():18.4f} {p_at_other.mean().item():19.6f} "
          f"{recall.item():11.4f} {fpr.item():8.6f} {loss_eot.item():10.4f} {loss_other.item():11.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
