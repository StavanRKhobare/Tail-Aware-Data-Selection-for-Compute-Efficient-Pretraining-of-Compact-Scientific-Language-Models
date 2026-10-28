"""Score every train document for quality, near-duplication and test leakage.

Outputs
  data/selection/quality.parquet   id, signals, quality_score, near_dup,
                                   contaminated, quality_pass
  reports/quality_report.json      rule hit rates, removals by topic group
  reports/figures/quality_*.png

Val / test abstracts and titles are used for one thing only: to find train
documents that are versions of the same paper (leakage). Those train documents
are excluded from every subset. No val / test statistic influences any score
or threshold.
"""
import json
import re
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import minhash
from quality import PENALTIES, SIGNALS, compute_signals, penalty, quality_score
from topics import assign_groups, category_groups


def _signals_chunk(abstracts):
    return [compute_signals(a) for a in abstracts]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main():
    config.ensure_dirs()
    cols = ["id", "title", "abstract", "primary_category"]
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet", columns=cols)
    train = train.sort_values("id").reset_index(drop=True)  # lowest id = cluster representative
    others = pd.concat(
        [pd.read_parquet(config.PROCESSED_DIR / f"{s}.parquet", columns=["id", "title", "abstract"]).assign(split=s)
         for s in ("val", "test")],
        ignore_index=True,
    )
    report = {"train_docs": len(train), "settings": {
        "QUALITY_MIN": config.QUALITY_MIN, "MINHASH_PERMS": config.MINHASH_PERMS,
        "MINHASH_BANDS": config.MINHASH_BANDS, "DUP_JACCARD": config.DUP_JACCARD,
        "penalties": [list(p) for p in PENALTIES]}}

    # 1. Heuristic signals and the graded quality score.
    abstracts = train["abstract"].tolist()
    chunks = [abstracts[i : i + 5_000] for i in range(0, len(abstracts), 5_000)]
    with Pool() as pool:
        sig = pd.DataFrame([s for part in pool.map(_signals_chunk, chunks) for s in part])
    df = pd.concat([train[["id", "primary_category"]], sig], axis=1)
    df["quality_score"] = [quality_score(r) for r in sig.to_dict("records")]
    print("signals done")

    # 2. MinHash over train + val + test abstracts.
    n_train = len(train)
    texts = abstracts + others["abstract"].tolist()
    sigs = minhash.signatures(texts, n_perm=config.MINHASH_PERMS, seed=config.SEED)
    pairs = minhash.candidate_pairs(sigs, config.MINHASH_BANDS)
    sim = minhash.similarity(sigs, pairs)
    edges = pairs[sim >= config.DUP_JACCARD]
    print(f"candidate pairs {len(pairs):,}, verified {len(edges):,}")

    # 2a. Near-duplicates inside train: keep the lowest id of each cluster.
    tt = edges[(edges[:, 0] < n_train) & (edges[:, 1] < n_train)]
    root = minhash.components(n_train, tt)
    df["near_dup"] = root != np.arange(n_train)
    df["near_dup_of"] = np.where(df["near_dup"], train["id"].to_numpy()[root], None)

    # 2b. Leakage: train documents that are versions of a val / test paper,
    # by abstract similarity or by an identical (normalized) title.
    cross = edges[(edges[:, 0] < n_train) & (edges[:, 1] >= n_train)]
    leak_text = np.zeros(n_train, dtype=bool)
    leak_text[cross[:, 0]] = True
    leak_title = train["title"].map(norm_title).isin(set(others["title"].map(norm_title))).to_numpy()
    df["contaminated"] = leak_text | leak_title
    leak_split = others["split"].to_numpy()[cross[:, 1] - n_train]
    report["near_dup"] = {
        "verified_pairs_train_train": int(len(tt)),
        "train_docs_marked_near_dup": int(df["near_dup"].sum()),
        "train_near_dup_clusters": int(len(np.unique(root[df["near_dup"].to_numpy()]))),
        "contaminated_train_docs": int(df["contaminated"].sum()),
        "contaminated_by_abstract": int(leak_text.sum()),
        "contaminated_by_title": int(leak_title.sum()),
        "leak_pairs_with_val": int((leak_split == "val").sum()),
        "leak_pairs_with_test": int((leak_split == "test").sum()),
    }
    ex = tt[:: max(len(tt) // 5, 1)][:5]
    report["near_dup"]["examples"] = [
        {"keep": train["id"][i], "drop": train["id"][j], "jaccard_est": round(float(s), 3),
         "keep_start": abstracts[i][:160], "drop_start": abstracts[j][:160]}
        for (i, j), s in zip(ex, minhash.similarity(sigs, ex))
    ]

    # 3. The quality filter used by the quality / diversity / tail_aware strategies.
    df["quality_pass"] = (df["quality_score"] >= config.QUALITY_MIN) & ~df["near_dup"]
    groups = assign_groups(df["primary_category"], category_groups(df["primary_category"]))
    df["group"] = groups

    rules = {}
    for name, d, a, b in PENALTIES:
        p = df[name].map(lambda v: penalty(v, d, a, b))
        rules[f"{name}:{d}"] = {"any_penalty_pct": round(100 * float((p > 0).mean()), 3),
                                "full_penalty_pct": round(100 * float((p >= 1).mean()), 3)}
    report["rule_hits"] = rules
    report["quality_score"] = {
        "pct_perfect": round(100 * float((df["quality_score"] == 1).mean()), 2),
        "pct_below_min": round(100 * float((df["quality_score"] < config.QUALITY_MIN).mean()), 2),
        "percentiles": {str(q): round(float(v), 4) for q, v in df["quality_score"].quantile([0.01, 0.05, 0.1, 0.25, 0.5]).items()},
    }
    by_group = df.groupby("group").agg(
        docs=("id", "size"),
        low_quality_pct=("quality_score", lambda s: 100 * (s < config.QUALITY_MIN).mean()),
        near_dup_pct=("near_dup", lambda s: 100 * s.mean()),
        contaminated_pct=("contaminated", lambda s: 100 * s.mean()),
        removed_by_filter_pct=("quality_pass", lambda s: 100 * (1 - s.mean())),
    ).round(3)
    report["by_group"] = by_group.to_dict(orient="index")
    report["quality_pool_docs"] = int((df["quality_pass"] & ~df["contaminated"]).sum())
    report["eligible_pool_docs"] = int((~df["contaminated"]).sum())

    signal_pcts = df[list(SIGNALS)].quantile([0.01, 0.5, 0.99])
    report["signal_percentiles"] = {c: {str(q): round(float(v), 4) for q, v in signal_pcts[c].items()} for c in SIGNALS}

    df.drop(columns="group").to_parquet(config.SELECTION_DIR / "quality.parquet", index=False)
    (config.REPORTS_DIR / "quality_report.json").write_text(json.dumps(report, indent=2))

    # Figures
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["quality_score"], bins=50)
    ax.axvline(config.QUALITY_MIN, color="k", ls="--", label=f"threshold {config.QUALITY_MIN}")
    ax.set_yscale("log")
    ax.set_title("Quality score (train), 1.0 = no heuristic issue")
    ax.set_xlabel("quality score")
    ax.set_ylabel("documents (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "quality_score_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    by_group[["low_quality_pct", "near_dup_pct", "contaminated_pct"]].loc[["head", "mid", "tail"]].plot.bar(ax=ax)
    ax.set_ylabel("% of train documents")
    ax.set_title("Who does the quality filter remove? (by topic group)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "quality_removed_by_group.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if k not in ("signal_percentiles", "settings")}, indent=2))


if __name__ == "__main__":
    main()
