"""Seed-level paired analysis of the five-seed Part 3 campaign.

Reads reports/5seed_experimental_results_runs.csv (produced by audit_results.py)
and reports, for each strategy pair:
  * per-seed differences,
  * mean / SD / SE over the five seeds,
  * a paired t interval (df = n-1),
  * a percentile bootstrap over seeds (resampling whole seeds).

The bootstrap over seeds is the relevant uncertainty for "would another training
seed give the same sign?". It is still only n = 5 and is reported as such.

No model is loaded and nothing is trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "reports" / "5seed_experimental_results_runs.csv"
OUT_CSV = ROOT / "reports" / "5seed_paired_seed_analysis.csv"
OUT_JSON = ROOT / "reports" / "5seed_paired_seed_analysis.json"

STRATEGIES = ("random", "quality", "diversity", "tail_aware", "tail_only")
PAIRS = (
    ("quality", "random"),
    ("tail_aware", "random"),
    ("tail_only", "tail_aware"),
    ("diversity", "random"),
    ("diversity", "quality"),
)
METRICS = ("overall_loss", "head_loss", "mid_loss", "tail_loss")
N_BOOT = 20000
BOOT_SEED = 20261110


def summarize(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    # two-sided 95% t interval, df = n-1
    try:
        from scipy import stats

        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        t_stat = mean / se if se else float("nan")
        p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2) if se else float("nan")
    except Exception:  # scipy is optional
        tcrit = float("nan")
        t_stat = mean / se if se else float("nan")
        p_value = float("nan")
    return {
        "n_seeds": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "t_stat": t_stat,
        "p_value_paired_t": p_value,
        "t_crit_975": tcrit,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
        "min_diff": float(values.min()),
        "max_diff": float(values.max()),
        "n_negative": int((values < 0).sum()),
        "n_positive": int((values > 0).sum()),
    }


def main() -> None:
    runs = pd.read_csv(RUNS)
    pivot = runs.set_index(["strategy", "seed"])
    seeds = sorted(int(s) for s in runs["seed"].unique())
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    for a, b in PAIRS:
        for metric in METRICS:
            diffs = np.array(
                [
                    float(pivot.loc[(a, s), metric]) - float(pivot.loc[(b, s), metric])
                    for s in seeds
                ]
            )
            stats = summarize(diffs)
            # Percentile bootstrap over whole seeds.
            boot = np.array(
                [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(N_BOOT)]
            )
            rows.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "metric": metric.replace("_loss", ""),
                    "per_seed_diffs": [round(float(x), 6) for x in diffs],
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()},
                    "boot_ci95_low": round(float(np.quantile(boot, 0.025)), 6),
                    "boot_ci95_high": round(float(np.quantile(boot, 0.975)), 6),
                    "boot_fraction_le_0": round(float((boot <= 0).mean()), 4),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)

    summary = (
        runs.groupby("strategy")
        .agg(
            seeds=("seed", "count"),
            overall_mean=("overall_loss", "mean"),
            overall_sd=("overall_loss", "std"),
            overall_min=("overall_loss", "min"),
            overall_max=("overall_loss", "max"),
            head_mean=("head_loss", "mean"),
            head_sd=("head_loss", "std"),
            mid_mean=("mid_loss", "mean"),
            mid_sd=("mid_loss", "std"),
            tail_mean=("tail_loss", "mean"),
            tail_sd=("tail_loss", "std"),
            tail_ppl_mean=("tail_ppl", "mean"),
        )
        .reindex(STRATEGIES)
        .reset_index()
    )
    summary.to_csv(ROOT / "reports" / "5seed_strategy_summary.csv", index=False)

    payload = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "unit": "training seed (whole checkpoint resampled)",
            "warning": "n=5; percentile intervals are coarse and describe these five seeds only.",
        },
        "strategy_summary": summary.to_dict(orient="records"),
        "paired": frame.to_dict(orient="records"),
    }
    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda o: int(o)
            if isinstance(o, np.integer)
            else float(o)
            if isinstance(o, np.floating)
            else str(o),
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print()
    cols = [
        "strategy_a", "strategy_b", "metric", "mean_diff", "sd_diff",
        "ci95_low", "ci95_high", "p_value_paired_t", "n_negative", "n_positive",
        "boot_ci95_low", "boot_ci95_high",
    ]
    print(frame[cols].to_string(index=False))
    print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
