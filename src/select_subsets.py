"""Build every training subset at the same token budget.

The design is a small factorial: quality filter (off / on) x topic sampling.

  strategy     candidate pool          topic sampling
  random       eligible                none (natural mix)
  quality      eligible + quality      none (natural mix)
  diversity    eligible + quality      equal weight per text theme (alpha 0)
  tail_aware   eligible + quality      primary category, temperature alpha 0.5  <- proposed
  tail_only    eligible                primary category, temperature alpha 0.5  (ablation)

"eligible" = every train document that is not a version of a val / test
paper. Every train document gets one random key (from the seed), and every
strategy takes documents in that same key order ("common random numbers").
So `quality` equals `random` except for the filtered documents and their
replacements: differences come from the strategy, not from sampling luck.

Outputs
  data/selection/subsets/<name>.parquet   chosen ids with topic / quality columns
  manifests/<name>.txt.gz                 sorted ids (committed, the reproducible record)
  reports/selection_report.json, reports/figures/selection_*.png
"""
import argparse
import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from sampling import random_keys, select
from topics import GROUPS

STRATEGIES = {
    "random": {"pool": "eligible", "topic": None},
    "quality": {"pool": "quality", "topic": None},
    "diversity": {"pool": "quality", "topic": "cluster", "alpha": config.DIVERSITY_ALPHA},
    "tail_aware": {"pool": "quality", "topic": "primary_category", "alpha": config.TAIL_ALPHA},
    "tail_only": {"pool": "eligible", "topic": "primary_category", "alpha": config.TAIL_ALPHA},
}


def load_pool():
    tr = pd.read_parquet(config.PROCESSED_DIR / "train.parquet", columns=["id", "v1_created"])
    tok = pd.read_parquet(config.SELECTION_DIR / "token_counts.parquet")
    tok = tok[tok["split"] == "train"][["id", "n_tokens"]]
    topics = pd.read_parquet(config.SELECTION_DIR / "topics.parquet")
    topics = topics[topics["split"] == "train"].drop(columns="split")
    qual = pd.read_parquet(config.SELECTION_DIR / "quality.parquet",
                           columns=["id", "quality_score", "near_dup", "contaminated", "quality_pass"])
    df = tr.merge(tok, on="id").merge(topics, on="id").merge(qual, on="id")
    assert len(df) == len(tr), "train ids do not line up across Part 2 inputs"
    df["tokens"] = df["n_tokens"] + 1  # + one end-of-text token per document
    df["year"] = df["v1_created"].dt.year
    return df.sort_values("id").reset_index(drop=True)


def effective_number(counts: pd.Series) -> float:
    """exp(entropy): how many equally-sized topics this mix is 'worth'."""
    p = counts[counts > 0] / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def shift(sub: pd.DataFrame, pool: pd.DataFrame, col: str) -> float:
    """Total variation distance of token mixes over `col`: 0 = same mix, 1 = disjoint."""
    a = pool.groupby(col)["tokens"].sum() / pool["tokens"].sum()
    b = sub.groupby(col)["tokens"].sum().reindex(a.index, fill_value=0) / sub["tokens"].sum()
    return float(0.5 * (a - b).abs().sum())


def describe(sub: pd.DataFrame, pool: pd.DataFrame) -> dict:
    g_sub = sub.groupby("group")["tokens"].sum().reindex(GROUPS, fill_value=0) / sub["tokens"].sum()
    g_pool = pool.groupby("group")["tokens"].sum().reindex(GROUPS, fill_value=0) / pool["tokens"].sum()
    cat = sub["primary_category"].value_counts()
    tail_cats = pool.loc[pool["group"] == "tail", "primary_category"].unique()
    return {
        "docs": len(sub),
        "tokens": int(sub["tokens"].sum()),
        "mean_tokens_per_doc": round(float(sub["tokens"].mean()), 1),
        "token_share_by_group": {g: round(float(v), 4) for g, v in g_sub.items()},
        "upweighting_vs_pool": {g: round(float(g_sub[g] / g_pool[g]), 3) for g in GROUPS},
        "categories_with_any_doc": int((cat > 0).sum()),
        "categories_with_50plus_docs": int((cat >= 50).sum()),
        "tail_categories_covered": f"{int(cat.reindex(tail_cats).fillna(0).gt(0).sum())}/{len(tail_cats)}",
        "effective_categories": round(effective_number(sub.groupby("primary_category")["tokens"].sum()), 2),
        "effective_themes": round(effective_number(sub.groupby("cluster")["tokens"].sum()), 2),
        "smallest_theme_tokens": int(sub.groupby("cluster")["tokens"].sum().reindex(range(config.N_CLUSTERS), fill_value=0).min()),
        "shift_vs_train_categories": round(shift(sub, pool, "primary_category"), 4),
        "shift_vs_train_themes": round(shift(sub, pool, "cluster"), 4),
        "mean_quality_score": round(float(sub["quality_score"].mean()), 4),
        "pct_low_quality": round(100 * float((sub["quality_score"] < config.QUALITY_MIN).mean()), 3),
        "near_dup_docs": int(sub["near_dup"].sum()),
        "pct_docs_2023_or_later": round(100 * float((sub["year"] >= 2023).mean()), 2),
        "median_year": int(sub["year"].median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED, help="random-key seed (use others for replicate subsets)")
    ap.add_argument("--budget", type=int, default=config.TOKEN_BUDGET)
    args = ap.parse_args()

    config.ensure_dirs()
    out_dir = config.SELECTION_DIR / "subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.seed == config.SEED else f"_seed{args.seed}"

    df = load_pool()
    df["rand"] = random_keys(len(df), args.seed)  # df is sorted by id, so keys are reproducible
    pools = {"eligible": df[~df["contaminated"]], "quality": df[~df["contaminated"] & df["quality_pass"]]}
    report = {
        "token_budget": args.budget,
        "seed": args.seed,
        "train_tokens_total": int(df["tokens"].sum()),
        "budget_as_pct_of_train": round(100 * args.budget / df["tokens"].sum(), 2),
        "pools": {k: {"docs": len(v), "tokens": int(v["tokens"].sum())} for k, v in pools.items()},
        "strategies": {k: {kk: vv for kk, vv in s.items()} for k, s in STRATEGIES.items()},
        "reference_full_train": describe(df, df),
        "subsets": {},
    }

    chosen = {}
    for name, spec in STRATEGIES.items():
        pool = pools[spec["pool"]]
        sub = select(pool, args.budget, topic=spec["topic"], alpha=spec.get("alpha", 1.0))
        assert sub["tokens"].sum() <= args.budget and sub["id"].is_unique
        chosen[name] = sub
        report["subsets"][name] = describe(sub, df)
        sub[["id", "tokens", "primary_category", "group", "cluster", "quality_score", "year"]].to_parquet(
            out_dir / f"{name}{suffix}.parquet", index=False)
        with gzip.open(config.MANIFEST_DIR / f"{name}{suffix}.txt.gz", "wt") as f:
            f.write("\n".join(sorted(sub["id"])) + "\n")
        print(name, json.dumps(report["subsets"][name]))

    ids = {k: set(v["id"]) for k, v in chosen.items()}
    report["overlap_jaccard"] = {a: {b: round(len(ids[a] & ids[b]) / len(ids[a] | ids[b]), 3) for b in ids} for a in ids}
    (config.REPORTS_DIR / f"selection_report{suffix}.json").write_text(json.dumps(report, indent=2))
    if suffix:
        return

    # Figures
    names = ["full train"] + list(STRATEGIES)
    shares = pd.DataFrame([report["reference_full_train"]["token_share_by_group"]] +
                          [report["subsets"][n]["token_share_by_group"] for n in STRATEGIES], index=names)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (100 * shares).plot.bar(stacked=True, ax=ax, color=["C0", "C1", "C2"])
    ax.set_ylabel("% of tokens")
    ax.set_title(f"Topic-group mix of each subset ({args.budget / 1e6:.0f}M tokens each)")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(shares["tail"]):
        ax.text(i, 101, f"tail {100 * v:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "selection_group_mix.png", dpi=150)
    plt.close(fig)

    order = df["primary_category"].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, style in zip(STRATEGIES, ["-", "--", ":", "-", "-."]):
        counts = chosen[name]["primary_category"].value_counts().reindex(order, fill_value=0)
        ax.plot(range(len(order)), counts.values + 0.5, style, label=name, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("primary category (ranked by size in full train)")
    ax.set_ylabel("documents selected (log)")
    ax.set_title("Documents per category: tail-aware sampling lifts the tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "selection_category_profile.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
