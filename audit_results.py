"""Read-only analysis of the existing Part 3 runs.

Computes comparable validation losses and perplexities for every checkpoint,
including head/mid/tail and per-primary-category breakdowns, plus paired
bootstrap intervals over validation documents. The model files are only read.

The bootstrap answers: "if the validation document sample were different, how
uncertain is this fixed-checkpoint difference?" It does NOT estimate variation
from training seeds.

Usage
-----
python audit_results.py                          # seeds 42, 43 -> experimental_results_*
python audit_results.py --seeds 42 43 44 45 46 --tag 5seed
                                                 # -> experimental_results_5seed_*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import GPT, GPTConfig  # noqa: E402

STRATEGIES = ("random", "quality", "diversity", "tail_aware", "tail_only")
SEEDS = (42, 43)
GROUPS = ("head", "mid", "tail")
BLOCK = 512
BATCH = 16
N_BOOT = 2000
BOOT_SEED = 20261110
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_difference(
    a_sum: np.ndarray,
    a_count: np.ndarray,
    b_sum: np.ndarray,
    b_count: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict:
    """Paired document bootstrap of token-weighted loss(a) - loss(b)."""
    a_s, a_c = a_sum[mask], a_count[mask]
    b_s, b_c = b_sum[mask], b_count[mask]
    n = len(a_s)
    observed = float(a_s.sum() / a_c.sum() - b_s.sum() / b_c.sum())
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a_s[idx].sum() / a_c[idx].sum() - b_s[idx].sum() / b_c[idx].sum()
    lower, upper = np.quantile(diffs, [0.025, 0.975])
    p_left = float((diffs <= 0).mean())
    p_right = float((diffs >= 0).mean())
    return {
        "observed_diff": observed,
        "boot_mean": float(diffs.mean()),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "two_sided_p": float(min(1.0, 2 * min(p_left, p_right))),
        "fraction_le_0": p_left,
        "n_docs_resampled": n,
        "n_tokens": int(a_c.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--tag",
        default=None,
        help="Prefix for output filenames, e.g. '5seed'. Default: no prefix.",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    reports = ROOT / "reports"
    prefix = f"{args.tag}_" if args.tag else ""

    def out(name: str) -> Path:
        return reports / f"{prefix}{name}"

    config = json.loads((ROOT / "runs" / "quality" / "seed42" / "config.json").read_text())
    assert config["model"]["block_size"] == BLOCK

    val = np.memmap(ROOT / "data" / "subsets" / "val.bin", dtype=np.uint16, mode="r")
    docs = pd.read_parquet(ROOT / "data" / "subsets" / "val_docs.parquet").sort_values(
        "start"
    ).reset_index(drop=True)
    ends = docs["end"].to_numpy(dtype=np.int64)
    assert ends[-1] == len(val)
    group_ids = docs["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(dtype=np.int64)
    categories = docs["primary_category"].to_numpy()
    n_windows = (len(val) - 1) // BLOCK
    starts = np.arange(n_windows, dtype=np.int64) * BLOCK
    print(
        f"device={DEVICE} docs={len(docs):,} val_tokens={len(val):,} "
        f"windows={n_windows:,} targets={n_windows*BLOCK:,}",
        flush=True,
    )

    run_rows = []
    group_rows = []
    category_rows = []
    per_doc: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    summaries: dict[tuple[str, int], dict] = {}

    for subset in STRATEGIES:
        for seed in seeds:
            run_dir = ROOT / "runs" / subset / f"seed{seed}"
            ckpt = torch.load(run_dir / "ckpt_best.pt", map_location="cpu", weights_only=False)
            cfg = GPTConfig(**ckpt["config"])
            model = GPT(cfg).to(DEVICE)
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()

            loss_sum = np.zeros(len(docs), dtype=np.float64)
            token_count = np.zeros(len(docs), dtype=np.int64)
            group_sum = np.zeros(len(GROUPS), dtype=np.float64)
            group_count = np.zeros(len(GROUPS), dtype=np.int64)
            total_sum = 0.0
            total_count = 0

            amp = torch.bfloat16 if DEVICE.type == "cuda" else None
            with torch.no_grad():
                for b0 in range(0, n_windows, BATCH):
                    idx = starts[b0 : b0 + BATCH]
                    x_np = np.stack([val[s : s + BLOCK] for s in idx]).astype(np.int64)
                    y_np = np.stack([val[s + 1 : s + 1 + BLOCK] for s in idx]).astype(np.int64)
                    x = torch.from_numpy(x_np).to(DEVICE)
                    y = torch.from_numpy(y_np).to(DEVICE)
                    if amp is not None:
                        with torch.autocast(device_type=DEVICE.type, dtype=amp):
                            logits, _ = model(x)
                    else:
                        logits, _ = model(x)
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y.reshape(-1),
                        reduction="none",
                    ).reshape(y.shape)
                    positions = (idx[:, None] + 1 + np.arange(BLOCK)[None, :]).ravel()
                    doc_idx = np.searchsorted(ends, positions, side="right")
                    g = group_ids[np.clip(doc_idx, 0, len(group_ids) - 1)]
                    flat = losses.reshape(-1).double().cpu().numpy()
                    loss_sum += np.bincount(doc_idx, weights=flat, minlength=len(docs))
                    token_count += np.bincount(doc_idx, minlength=len(docs)).astype(np.int64)
                    group_sum += np.bincount(g, weights=flat, minlength=len(GROUPS))
                    group_count += np.bincount(g, minlength=len(GROUPS)).astype(np.int64)
                    total_sum += float(flat.sum())
                    total_count += flat.size

            overall = total_sum / total_count
            if abs(overall - float(ckpt["val_loss"])) > 5e-4:
                print(
                    f"WARNING {subset}/seed{seed}: recomputed {overall:.8f} vs "
                    f"checkpoint {float(ckpt['val_loss']):.8f}",
                    file=sys.stderr,
                )
            per_doc[(subset, seed)] = (loss_sum, token_count)
            summaries[(subset, seed)] = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )

            row = {
                "strategy": subset,
                "seed": seed,
                "overall_loss": overall,
                "overall_ppl": float(np.exp(overall)),
                "tokens": total_count,
                "checkpoint_val_loss": float(ckpt["val_loss"]),
                "checkpoint_step": int(ckpt["step"]),
            }
            for i, group in enumerate(GROUPS):
                loss = group_sum[i] / group_count[i]
                row[f"{group}_loss"] = float(loss)
                row[f"{group}_ppl"] = float(np.exp(loss))
                row[f"{group}_tokens"] = int(group_count[i])
            run_rows.append(row)

            # Per-document aggregation -> per-category losses.
            doc_frame = pd.DataFrame(
                {
                    "primary_category": categories,
                    "group": docs["group"].to_numpy(),
                    "loss_sum": loss_sum,
                    "tokens": token_count,
                }
            )
            cat = doc_frame.groupby(["primary_category", "group"], as_index=False).agg(
                loss_sum=("loss_sum", "sum"), tokens=("tokens", "sum")
            )
            cat = cat[cat["tokens"] > 0].copy()
            cat["loss"] = cat["loss_sum"] / cat["tokens"]
            cat["ppl"] = np.exp(cat["loss"])
            cat["strategy"] = subset
            cat["seed"] = seed
            category_rows.append(cat.drop(columns="loss_sum"))
            group_rows.append(
                pd.DataFrame(
                    [
                        {
                            "strategy": subset,
                            "seed": seed,
                            "group": group,
                            "loss": row[f"{group}_loss"],
                            "ppl": row[f"{group}_ppl"],
                            "tokens": row[f"{group}_tokens"],
                        }
                        for group in GROUPS
                    ]
                )
            )
            print(
                f"{subset:11} seed{seed} overall={overall:.6f} "
                + " ".join(f"{g}={row[f'{g}_loss']:.6f}" for g in GROUPS),
                flush=True,
            )
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    runs = pd.DataFrame(run_rows).sort_values(["strategy", "seed"]).reset_index(drop=True)
    groups_out = pd.concat(group_rows, ignore_index=True)
    categories_out = pd.concat(category_rows, ignore_index=True)

    strategy_summary = (
        runs.groupby("strategy")
        .agg(
            seeds=("seed", "count"),
            overall_mean=("overall_loss", "mean"),
            overall_std=("overall_loss", "std"),
            overall_ppl_mean=("overall_ppl", "mean"),
            head_mean=("head_loss", "mean"),
            head_std=("head_loss", "std"),
            mid_mean=("mid_loss", "mean"),
            mid_std=("mid_loss", "std"),
            tail_mean=("tail_loss", "mean"),
            tail_std=("tail_loss", "std"),
            tail_ppl_mean=("tail_ppl", "mean"),
        )
        .reindex(STRATEGIES)
        .reset_index()
    )

    # Paired per-seed differences (same initialization seed, different data).
    pairs = (
        ("quality", "random"),
        ("tail_aware", "random"),
        ("tail_only", "tail_aware"),
        ("diversity", "random"),
        ("diversity", "quality"),
    )
    diff_rows = []
    pivot = runs.set_index(["strategy", "seed"])
    for a, b in pairs:
        for seed in seeds:
            row_a, row_b = pivot.loc[(a, seed)], pivot.loc[(b, seed)]
            diff_rows.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "seed": seed,
                    "overall_diff": row_a["overall_loss"] - row_b["overall_loss"],
                    "head_diff": row_a["head_loss"] - row_b["head_loss"],
                    "mid_diff": row_a["mid_loss"] - row_b["mid_loss"],
                    "tail_diff": row_a["tail_loss"] - row_b["tail_loss"],
                    "tail_ppl_ratio": row_a["tail_ppl"] / row_b["tail_ppl"],
                }
            )
    diffs = pd.DataFrame(diff_rows)

    # Paired validation-document bootstrap, separately for each training seed.
    rng = np.random.default_rng(BOOT_SEED)
    masks = {
        "overall": np.ones(len(docs), dtype=bool),
        **{g: docs["group"].to_numpy() == g for g in GROUPS},
    }
    boot_rows = []
    for a, b in pairs:
        for seed in seeds:
            a_sum, a_cnt = per_doc[(a, seed)]
            b_sum, b_cnt = per_doc[(b, seed)]
            for scope, mask in masks.items():
                result = bootstrap_difference(a_sum, a_cnt, b_sum, b_cnt, mask, rng)
                boot_rows.append(
                    {
                        "strategy_a": a,
                        "strategy_b": b,
                        "seed": seed,
                        "scope": scope,
                        **result,
                    }
                )
    boot = pd.DataFrame(boot_rows)

    # Validation-loss trajectories from the existing logs.
    curve_rows = []
    run_summaries = []
    for subset in STRATEGIES:
        for seed in seeds:
            summary = summaries[(subset, seed)]
            run_summaries.append(
                {
                    "strategy": subset,
                    "seed": seed,
                    "step": summary["step"],
                    "tokens_seen": summary["tokens_seen"],
                    "final_val_loss": summary["final_val_loss"],
                    "final_val_ppl": summary["final_val_ppl"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_step": summary["best_step"],
                    "wall_time_s": summary["wall_time_s"],
                    "tokens_per_s": summary["tokens_per_s"],
                    "peak_mem_mb": summary["peak_mem_mb"],
                    "model_params": summary["model_params"],
                }
            )
            path = ROOT / "runs" / subset / f"seed{seed}" / "metrics.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("val_loss") is None:
                    continue
                curve_rows.append(
                    {
                        "strategy": subset,
                        "seed": seed,
                        "step": record["step"],
                        "tokens": record["tokens"],
                        "val_loss": record["val_loss"],
                        "val_ppl": record["val_ppl"],
                        "elapsed_s": record["elapsed_s"],
                        "tokens_per_s": record["tokens_per_s"],
                        "peak_mem_mb": record["peak_mem_mb"],
                    }
                )
    curves = pd.DataFrame(curve_rows)
    curve_summary = []
    for (subset, seed), part in curves.groupby(["strategy", "seed"]):
        part = part.sort_values("tokens")
        final = part.iloc[-1]
        at_30m = part[part["tokens"] <= 30_000_000].iloc[-1]
        at_50m = part[part["tokens"] <= 50_000_000].iloc[-1]
        at_40m = part[part["tokens"] <= 40_000_000].iloc[-1]
        curve_summary.append(
            {
                "strategy": subset,
                "seed": seed,
                "loss_30m": at_30m["val_loss"],
                "loss_40m": at_40m["val_loss"],
                "loss_50m": at_50m["val_loss"],
                "loss_final": final["val_loss"],
                "improvement_30m_to_final": at_30m["val_loss"] - final["val_loss"],
                "improvement_50m_to_final": at_50m["val_loss"] - final["val_loss"],
                "improvement_per_10m_last": 10 * (at_50m["val_loss"] - final["val_loss"]) / 10,
                "final_step": int(final["step"]),
                "final_tokens": int(final["tokens"]),
            }
        )
    curve_summary = pd.DataFrame(curve_summary)

    # Per-category paired effects, averaged over the two training seeds.
    cat_mean = (
        categories_out.groupby(["primary_category", "group", "strategy"])["loss"]
        .mean()
        .unstack("strategy")
    )
    cat_mean["tail_aware_minus_random"] = cat_mean["tail_aware"] - cat_mean["random"]
    cat_mean["quality_minus_random"] = cat_mean["quality"] - cat_mean["random"]
    cat_mean["tail_only_minus_tail_aware"] = cat_mean["tail_only"] - cat_mean["tail_aware"]
    cat_mean["diversity_minus_random"] = cat_mean["diversity"] - cat_mean["random"]
    cat_mean = cat_mean.reset_index()
    cat_mean.to_csv(out("experimental_results_per_category.csv"), index=False)

    categories_out.to_csv(
        out("experimental_results_category_by_seed.csv"), index=False
    )
    runs.to_csv(out("experimental_results_runs.csv"), index=False)
    groups_out.to_csv(out("experimental_results_groups.csv"), index=False)
    strategy_summary.to_csv(
        out("experimental_results_strategy_summary.csv"), index=False
    )
    diffs.to_csv(out("experimental_results_paired_diffs.csv"), index=False)
    boot.to_csv(out("experimental_results_bootstrap.csv"), index=False)
    curves.to_csv(out("experimental_results_val_curves.csv"), index=False)
    curve_summary.to_csv(
        out("experimental_results_val_curve_summary.csv"), index=False
    )
    pd.DataFrame(run_summaries).to_csv(
        out("experimental_results_run_summaries.csv"), index=False
    )

    payload = {
        "device": str(DEVICE),
        "block_size": BLOCK,
        "seeds": list(seeds),
        "windows": n_windows,
        "target_tokens": int(n_windows * BLOCK),
        "val_docs": len(docs),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "unit": "validation document (paired, token-weighted loss)",
            "warning": "Estimates validation-sampling uncertainty for fixed checkpoints; "
            "not training-seed uncertainty.",
        },
        "primary_metric_note": "PART3.md defines final validation loss at 60M tokens as the "
        "primary metric. Head/mid/tail and per-category results are diagnostic and were not "
        "preregistered as primary outcomes.",
        "strategy_summary": strategy_summary.to_dict(orient="records"),
        "seed_differences": diffs.to_dict(orient="records"),
        "curve_summary": curve_summary.to_dict(orient="records"),
        "top_tail_categories_tail_aware_minus_random": cat_mean[
            (cat_mean["group"] == "tail") & (cat_mean["tail_aware_minus_random"].notna())
        ].sort_values("tail_aware_minus_random").head(15).to_dict(orient="records"),
        "top_head_categories_tail_aware_minus_random": cat_mean[
            (cat_mean["group"] == "head") & (cat_mean["tail_aware_minus_random"].notna())
        ].sort_values("tail_aware_minus_random").head(15).to_dict(orient="records"),
    }
    (out("experimental_results_audit.json")).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nRUNS")
    print(runs.to_string(index=False))
    print("\nSTRATEGY SUMMARY")
    print(strategy_summary.to_string(index=False))
    print("\nPAIRED DIFFERENCES")
    print(diffs.to_string(index=False))
    print("\nCURVE SUMMARY")
    print(curve_summary.to_string(index=False))
    print("\nBOOTSTRAP (overall only)")
    print(
        boot[boot["scope"] == "overall"][
            ["strategy_a", "strategy_b", "seed", "observed_diff", "ci95_low", "ci95_high", "two_sided_p"]
        ].to_string(index=False)
    )
    print(f"\nwrote reports/{prefix}experimental_results_*.{{json,csv}}")


if __name__ == "__main__":
    main()
