"""Read-only audit of the ten Part 3 runs: configs, loss curves, stability."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RUNS = Path("runs")
subsets = ["random", "quality", "diversity", "tail_aware", "tail_only"]
seeds = [42, 43]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


print("=" * 90)
print("RUN CONFIG / SUMMARY COMPARISON")
print("=" * 90)
rows = []
for s in subsets:
    for seed in seeds:
        d = RUNS / s / f"seed{seed}"
        cfg = load_json(d / "config.json")
        summ = load_json(d / "summary.json")
        rows.append((s, seed, cfg, summ))
        print(
            f"{s:11} seed{seed} | params={summ['model_params']/1e6:.3f}M "
            f"tokens={summ['tokens_seen']} steps={summ['step']} "
            f"final={summ['final_val_loss']:.4f} best={summ['best_val_loss']:.4f} "
            f"best_step={summ['best_step']} wall={summ['wall_time_s']:.1f}s"
        )

print("\n" + "=" * 90)
print("CONFIG IDENTITY CHECK (everything except subset/seed must match)")
print("=" * 90)
base = rows[0][2]
keys_model = list(base["model"].keys())
keys_train = [k for k in base["train"].keys() if k not in ("subset", "seed", "out_dir")]
print("model keys:", keys_model)
print("train keys compared:", keys_train)
mismatches = []
for s, seed, cfg, _ in rows:
    for k in keys_model:
        if cfg["model"][k] != base["model"][k]:
            mismatches.append((s, seed, f"model.{k}", cfg["model"][k], base["model"][k]))
    for k in keys_train:
        if cfg["train"].get(k) != base["train"].get(k):
            mismatches.append((s, seed, f"train.{k}", cfg["train"].get(k), base["train"].get(k)))
    if cfg["data"]["vocab_size"] != 16384 or cfg["data"]["eot_id"] != 0 or cfg["data"]["pad_id"] != 1:
        mismatches.append((s, seed, "data special ids", cfg["data"], "expected 16384/0/1"))
print("mismatches:", mismatches if mismatches else "NONE")

print("\n" + "=" * 90)
print("PER-RUN LOSS CURVE / STABILITY")
print("=" * 90)
for s, seed, cfg, summ in rows:
    d = RUNS / s / f"seed{seed}"
    recs = [json.loads(line) for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    evals = [r for r in recs if r.get("val_loss") is not None]
    train = [r for r in recs if r.get("train_loss") is not None]
    gnorms = np.array([r["grad_norm"] for r in train if r.get("grad_norm") is not None], dtype=float)
    tl = np.array([r["train_loss"] for r in train], dtype=float)
    vl = np.array([r["val_loss"] for r in evals], dtype=float)
    steps = np.array([r["step"] for r in evals])
    imin = int(vl.argmin())
    print(f"\n{s} seed{seed}")
    print(f"  logged rows={len(recs)} eval_points={len(evals)} train_points={len(train)}")
    print(f"  train loss: first={tl[0]:.4f} min={tl.min():.4f} last5_mean={tl[-5:].mean():.4f} "
          f"max_jump={np.abs(np.diff(tl)).max():.4f}")
    print(f"  val   loss: first={vl[0]:.4f} min={vl.min():.4f}@{steps[imin]} last={vl[-1]:.4f} "
          f"increase_from_min={vl[-1]-vl.min():+.5f}")
    print(f"  grad norm : min={gnorms.min():.3f} mean={gnorms.mean():.3f} max={gnorms.max():.3f} "
          f"nonzero={int((gnorms>0).sum())}/{len(gnorms)}")
    print(f"  lr        : first={train[0]['lr']:.3e} last={train[-1]['lr']:.3e}")
    print(f"  last 4 val: {[(int(r['step']), round(r['val_loss'],4)) for r in evals[-4:]]}")

print("\n" + "=" * 90)
print("TRAIN/VAL TOKEN COUNTS PER SUBSET")
print("=" * 90)
for s in subsets:
    cfg = load_json(RUNS / s / "seed42" / "config.json")
    print(f"{s:11} train_tokens={cfg['data']['train_tokens']:,} val_tokens={cfg['data']['val_tokens']:,} "
          f"train_bin={cfg['data']['train_bin']}")

print("\n" + "=" * 90)
print("SEED PAIRING")
print("=" * 90)
for s in subsets:
    a = load_json(RUNS / s / "seed42" / "config.json")["data"]["train_bin"]
    b = load_json(RUNS / s / "seed43" / "config.json")["data"]["train_bin"]
    print(f"{s:11} seed42 and seed43 use same train.bin: {a == b} ({a})")

print("\n" + "=" * 90)
print("METRIC vs REPORT CONSISTENCY")
print("=" * 90)
report = load_json("reports/part3_report.json")
for s in subsets:
    finals = [load_json(RUNS / s / f"seed{seed}" / "summary.json")["final_val_loss"] for seed in seeds]
    mean = float(np.mean(finals))
    print(f"{s:11} recomputed={mean:.6f} report={report[s]['mean_final_val_loss']:.6f} "
          f"diff={mean-report[s]['mean_final_val_loss']:+.2e}")
