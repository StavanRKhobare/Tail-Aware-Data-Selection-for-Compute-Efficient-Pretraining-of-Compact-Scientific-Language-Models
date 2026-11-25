"""Read-only audit of the five selected training subsets.

Why this script exists
----------------------
The selection parquet files (`data/selection/subsets/*.parquet`) and the
per-document score files (`quality.parquet`, `token_counts.parquet`) are not
present in this checkout. The committed manifests and tokenized `train.bin`
files are present, so this audit reconstructs the selected document ids and
exact token counts from those canonical artifacts. It never writes into
`data/`, `runs/`, or `manifests/`.

Outputs (all new; existing reports are not touched)
  reports/selection_audit.json
  reports/selection_audit_summary.csv
  reports/selection_audit_groups.csv
  reports/selection_audit_categories.csv
  reports/selection_audit_themes.csv
  reports/selection_audit_lengths.csv
  reports/selection_audit_quality.csv
  reports/selection_audit_overlap.csv

Definitions are the project's own:
  * group: head >= 10,000 train docs in the primary category, mid >= 1,000,
    tail < 1,000 (config.HEAD_MIN_DOCS / MID_MIN_DOCS).
  * theme: field `cluster` (0-99), hierarchical k-means over TF-IDF/SVD text
    features, fit on train only (src/build_topics.py).
  * quality: graded product of rule penalties from src/quality.py; lower is
    worse, 1.0 means no heuristic issue.
  * tokens: tokenizer ids plus one EOT per document, matching export_tokens.py.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from quality import compute_signals, quality_score  # noqa: E402

STRATEGIES = ("random", "quality", "diversity", "tail_aware", "tail_only")
GROUPS = ("head", "mid", "tail")
N_FINE = config.N_FINE_CLUSTERS
N_THEMES = config.N_CLUSTERS
EOT = None


def _score_chunk(abstracts: list[str]) -> list[float]:
    return [quality_score(compute_signals(text)) for text in abstracts]


def score_quality(abstracts: list[str], workers: int) -> np.ndarray:
    chunks = [abstracts[i : i + 5_000] for i in range(0, len(abstracts), 5_000)]
    if workers <= 1:
        values = [value for chunk in chunks for value in _score_chunk(chunk)]
    else:
        with mp.Pool(processes=workers) as pool:
            values = [value for part in pool.map(_score_chunk, chunks) for value in part]
    return np.asarray(values, dtype=np.float64)


def token_lengths(texts: list[str], tok: Tokenizer) -> np.ndarray:
    lengths: list[int] = []
    for i in range(0, len(texts), 10_000):
        lengths.extend(len(enc.ids) + 1 for enc in tok.encode_batch(texts[i : i + 10_000]))
    return np.asarray(lengths, dtype=np.int64)


def effective_number(counts: pd.Series) -> float:
    values = counts[counts > 0].to_numpy(dtype=np.float64)
    if values.size == 0:
        return 0.0
    p = values / values.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def top_share(counts: pd.Series) -> float:
    total = float(counts.sum())
    return float(counts.max() / total) if total else 0.0


def theme_metrics(frame: pd.DataFrame, n_themes: int) -> dict:
    docs = frame.groupby("cluster").size().reindex(range(n_themes), fill_value=0)
    tokens = frame.groupby("cluster")["tokens"].sum().reindex(range(n_themes), fill_value=0)
    return {
        "themes_covered": int((docs > 0).sum()),
        "themes_total": n_themes,
        "theme_docs_effective": round(effective_number(docs), 3),
        "theme_tokens_effective": round(effective_number(tokens), 3),
        "top_theme_doc_share": round(top_share(docs), 4),
        "top_theme_token_share": round(top_share(tokens), 4),
        "smallest_theme_tokens": int(tokens.min()),
        "median_theme_tokens": int(tokens.median()),
        "theme_token_cv": round(float(tokens.std(ddof=0) / tokens.mean()), 4) if tokens.mean() else None,
        "empty_themes": int((docs == 0).sum()),
        "fine_clusters_covered": int(frame["fine_cluster"].nunique()),
        "fine_clusters_total": N_FINE,
    }


def category_metrics(frame: pd.DataFrame, all_categories: list[str]) -> dict:
    docs = frame["primary_category"].value_counts().reindex(all_categories, fill_value=0)
    tokens = frame.groupby("primary_category")["tokens"].sum().reindex(all_categories, fill_value=0)
    return {
        "categories_covered": int((docs > 0).sum()),
        "categories_total": len(all_categories),
        "category_docs_effective": round(effective_number(docs), 3),
        "category_tokens_effective": round(effective_number(tokens), 3),
        "top_category_doc_share": round(top_share(docs), 4),
        "top_category_token_share": round(top_share(tokens), 4),
        "categories_with_50plus_docs": int((docs >= 50).sum()),
    }


def length_metrics(tokens: pd.Series) -> dict:
    q = tokens.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "token_len_mean": round(float(tokens.mean()), 2),
        "token_len_std": round(float(tokens.std(ddof=0)), 2),
        "token_len_min": int(tokens.min()),
        "token_len_p10": int(q.loc[0.1]),
        "token_len_p25": int(q.loc[0.25]),
        "token_len_median": int(q.loc[0.5]),
        "token_len_p75": int(q.loc[0.75]),
        "token_len_p90": int(q.loc[0.9]),
        "token_len_p99": int(q.loc[0.99]),
        "token_len_max": int(tokens.max()),
    }


def quality_metrics(frame: pd.DataFrame) -> dict:
    scores = frame["quality_score"]
    return {
        "quality_missing": int(scores.isna().sum()),
        "quality_mean": round(float(scores.mean()), 5),
        "quality_median": round(float(scores.median()), 5),
        "quality_p10": round(float(scores.quantile(0.1)), 5),
        "quality_p90": round(float(scores.quantile(0.9)), 5),
        "quality_pct_perfect": round(100 * float((scores == 1.0).mean()), 3),
        "quality_pct_below_min": round(100 * float((scores < config.QUALITY_MIN).mean()), 3),
        "quality_pct_below_0_9": round(100 * float((scores < 0.9).mean()), 3),
    }


def group_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    total_docs = len(frame)
    total_tokens = int(frame["tokens"].sum())
    rows = []
    for group in GROUPS:
        part = frame[frame["group"] == group]
        rows.append(
            {
                "strategy": name,
                "group": group,
                "docs": len(part),
                "doc_share": len(part) / total_docs if total_docs else 0.0,
                "tokens": int(part["tokens"].sum()),
                "token_share": int(part["tokens"].sum()) / total_tokens if total_tokens else 0.0,
                "mean_quality": float(part["quality_score"].mean()) if len(part) else None,
            }
        )
    return pd.DataFrame(rows)


def strategy_metrics(name: str, frame: pd.DataFrame, full: pd.DataFrame, all_categories: list[str]) -> dict:
    docs = len(frame)
    tokens = int(frame["tokens"].sum())
    full_docs = len(full)
    full_tokens = int(full["tokens"].sum())
    result = {
        "strategy": name,
        "docs": docs,
        "unique_docs": int(frame["id"].nunique()),
        "tokens": tokens,
        "mean_tokens_per_doc": round(float(frame["tokens"].mean()), 2),
        "doc_share_of_train": round(docs / full_docs, 6),
        "token_share_of_train": round(tokens / full_tokens, 6),
        "exact_duplicate_text_docs": int(frame["text"].duplicated().sum()),
        "exact_duplicate_text_pct": round(100 * float(frame["text"].duplicated().mean()), 4),
        "mean_quality_score": round(float(frame["quality_score"].mean()), 5),
        "pct_low_quality": round(100 * float((frame["quality_score"] < config.QUALITY_MIN).mean()), 3),
        **{k: v for k, v in category_metrics(frame, all_categories).items()},
        **{k: v for k, v in theme_metrics(frame, N_THEMES).items()},
        **{k: v for k, v in length_metrics(frame["tokens"]).items()},
        **{k: v for k, v in quality_metrics(frame).items()},
    }
    for group in GROUPS:
        part = frame[frame["group"] == group]
        result[f"group_{group}_docs"] = len(part)
        result[f"group_{group}_doc_share"] = float(len(part) / docs) if docs else 0.0
        result[f"group_{group}_tokens"] = int(part["tokens"].sum())
        result[f"group_{group}_token_share"] = (
            int(part["tokens"].sum()) / tokens if tokens else 0.0
        )
    return result


def main() -> None:
    config.ensure_dirs()
    reports = ROOT / "reports"

    missing = [
        path
        for path in (
            "data/selection/quality.parquet",
            "data/selection/token_counts.parquet",
            "data/selection/subsets/random.parquet",
        )
        if not (ROOT / path).exists()
    ]

    train = pd.read_parquet(
        config.PROCESSED_DIR / "train.parquet",
        columns=["id", "title", "abstract", "text", "primary_category", "n_words"],
    ).sort_values("id").reset_index(drop=True)
    topics = pd.read_parquet(config.SELECTION_DIR / "topics.parquet")
    topics = topics[topics["split"] == "train"].drop(columns="split")
    train = train.merge(
        topics[["id", "group", "fine_cluster", "cluster"]], on="id", how="left", validate="one_to_one"
    )
    assert train["group"].notna().all(), "some train docs are missing topic labels"
    assert train["id"].is_unique

    print(f"train docs={len(train):,}; recomputing quality scores...", flush=True)
    workers = max(1, min(8, (mp.cpu_count() or 1)))
    train["quality_score"] = score_quality(train["abstract"].tolist(), workers)

    print("recomputing token lengths...", flush=True)
    tok = Tokenizer.from_file(str(config.TOKENIZER_DIR / "tokenizer.json"))
    train["tokens"] = token_lengths(train["text"].tolist(), tok)
    text_tokens = int((train["tokens"] - 1).sum())
    total_tokens = int(train["tokens"].sum())
    print(f"train tokens={total_tokens:,} (text {text_tokens:,} + {len(train):,} EOT)", flush=True)

    # Reconstruct exact selected token counts from train.bin order. export_tokens
    # writes a seeded permutation of the sorted manifest ids, one EOT per doc.
    by_id = train.set_index("id")["tokens"]
    frames: dict[str, pd.DataFrame] = {}
    verification: dict[str, dict] = {}
    for name in STRATEGIES:
        ids = gzip.open(config.MANIFEST_DIR / f"{name}.txt.gz", "rt").read().split()
        order = np.random.default_rng(config.SEED).permutation(len(ids))
        arr = np.memmap(config.SUBSETS_DIR / name / "train.bin", dtype=np.uint16, mode="r")
        eot_positions = np.flatnonzero(arr == 0)
        lengths = np.diff(np.r_[-1, eot_positions])
        assert len(eot_positions) == len(ids), f"{name}: EOT count != manifest length"
        assert int(lengths.sum()) == len(arr)

        ordered_ids = [ids[i] for i in order]
        bin_lengths = pd.Series(lengths, index=ordered_ids, dtype=np.int64)
        frame = train[train["id"].isin(ids)].copy()
        frame["tokens_bin"] = frame["id"].map(bin_lengths)
        assert frame["tokens_bin"].notna().all(), f"{name}: missing bin length for a doc"
        mismatches = int((frame["tokens"] != frame["tokens_bin"]).sum())
        assert mismatches == 0, f"{name}: {mismatches} token-length mismatches"
        verification[name] = {
            "manifest_docs": len(ids),
            "unique_manifest_ids": len(set(ids)),
            "bin_docs": int(len(eot_positions)),
            "bin_tokens": int(len(arr)),
            "tokens_recomputed": int(frame["tokens"].sum()),
            "token_length_mismatches": mismatches,
            "last_token_is_eot": int(arr[-1]) == 0,
            "double_eot_pairs": int((((arr[:-1] == 0) & (arr[1:] == 0))).sum()),
        }
        frames[name] = frame

    all_categories = sorted(train["primary_category"].unique())
    full = train

    summary_rows = []
    group_rows = []
    category_rows = []
    theme_rows = []
    length_rows = []
    quality_rows = []
    score_by_group = []

    for name in ("full_train",) + STRATEGIES:
        frame = full if name == "full_train" else frames[name]
        label = "full_train" if name == "full_train" else name
        row = strategy_metrics(label, frame, full, all_categories)
        summary_rows.append(row)
        group_counts = group_table(frame, label)
        if name != "full_train":
            group_counts["reference_doc_share"] = group_counts["group"].map(
                full["group"].value_counts(normalize=True)
            )
            group_counts["reference_token_share"] = group_counts["group"].map(
                full.groupby("group")["tokens"].sum() / full["tokens"].sum()
            )
            group_counts["doc_upweight_vs_train"] = (
                group_counts["doc_share"] / group_counts["reference_doc_share"]
            )
            group_counts["token_upweight_vs_train"] = (
                group_counts["token_share"] / group_counts["reference_token_share"]
            )
        group_rows.append(group_counts)

        cats = frame.groupby("primary_category").agg(
            docs=("id", "size"), tokens=("tokens", "sum"), mean_quality=("quality_score", "mean")
        )
        cats = cats.reindex(all_categories, fill_value=0)
        full_cats = full.groupby("primary_category").agg(
            train_docs=("id", "size"), train_tokens=("tokens", "sum")
        ).reindex(all_categories, fill_value=0)
        cat_table = cats.join(full_cats)
        cat_table["strategy"] = label
        cat_table["group"] = cat_table.index.map(
            full.drop_duplicates("primary_category").set_index("primary_category")["group"]
        )
        cat_table["doc_share"] = cat_table["docs"] / max(len(frame), 1)
        cat_table["token_share"] = cat_table["tokens"] / max(int(frame["tokens"].sum()), 1)
        cat_table["train_doc_share"] = cat_table["train_docs"] / len(full)
        cat_table["train_token_share"] = cat_table["train_tokens"] / int(full["tokens"].sum())
        category_rows.append(cat_table.reset_index(names="primary_category"))

        themes = frame.groupby("cluster").agg(
            docs=("id", "size"), tokens=("tokens", "sum"), mean_quality=("quality_score", "mean")
        ).reindex(range(N_THEMES), fill_value=0)
        themes["strategy"] = label
        themes["theme"] = themes.index
        themes["doc_share"] = themes["docs"] / max(len(frame), 1)
        themes["token_share"] = themes["tokens"] / max(int(frame["tokens"].sum()), 1)
        theme_rows.append(themes.reset_index(drop=True))

        length_rows.append({"strategy": label, **length_metrics(frame["tokens"])})
        quality_rows.append({"strategy": label, **quality_metrics(frame)})
        for group in GROUPS:
            part = frame[frame["group"] == group]
            score_by_group.append(
                {
                    "strategy": label,
                    "group": group,
                    "docs": len(part),
                    "mean_quality": float(part["quality_score"].mean()) if len(part) else None,
                    "pct_below_min": (
                        float((part["quality_score"] < config.QUALITY_MIN).mean()) if len(part) else None
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    groups_out = pd.concat(group_rows, ignore_index=True)
    categories_out = pd.concat(category_rows, ignore_index=True)
    themes_out = pd.concat(theme_rows, ignore_index=True)
    lengths_out = pd.DataFrame(length_rows)
    quality_out = pd.DataFrame(quality_rows)
    quality_group_out = pd.DataFrame(score_by_group)

    # Pairwise overlap, reported separately for documents and tokens.
    overlap_rows = []
    for a in STRATEGIES:
        for b in STRATEGIES:
            sa, sb = set(frames[a]["id"]), set(frames[b]["id"])
            inter = sa & sb
            union = sa | sb
            tokens_a = frames[a].set_index("id")["tokens"]
            tokens_b = frames[b].set_index("id")["tokens"]
            overlap_rows.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "unique_a": len(sa),
                    "unique_b": len(sb),
                    "intersection_docs": len(inter),
                    "union_docs": len(union),
                    "jaccard_docs": len(inter) / len(union) if union else None,
                    "intersection_tokens_a": int(tokens_a.reindex(inter).sum()),
                    "jaccard_tokens_asym": (
                        int(tokens_a.reindex(inter).sum()) / int(tokens_a.sum())
                        if tokens_a.sum()
                        else None
                    ),
                }
            )
    overlap = pd.DataFrame(overlap_rows)

    # Explicit comparisons used in the report.
    def row(name: str) -> pd.Series:
        return summary[summary["strategy"] == name].iloc[0]

    q = row("quality")
    r = row("random")
    ta = row("tail_aware")
    to = row("tail_only")
    dv = row("diversity")
    comparisons = {
        "quality_vs_random": {
            "mean_quality_diff": float(q["mean_quality_score"] - r["mean_quality_score"]),
            "pct_low_quality_diff": float(q["pct_low_quality"] - r["pct_low_quality"]),
            "tail_token_share_diff": float(q["group_tail_token_share"] - r["group_tail_token_share"]),
        },
        "tail_aware_vs_random": {
            "tail_doc_share_diff": float(ta["group_tail_doc_share"] - r["group_tail_doc_share"]),
            "tail_token_share_diff": float(ta["group_tail_token_share"] - r["group_tail_token_share"]),
            "head_token_share_diff": float(ta["group_head_token_share"] - r["group_head_token_share"]),
        },
        "tail_only_vs_tail_aware": {
            "tail_doc_share_diff": float(to["group_tail_doc_share"] - ta["group_tail_doc_share"]),
            "tail_token_share_diff": float(to["group_tail_token_share"] - ta["group_tail_token_share"]),
            "mean_quality_diff": float(to["mean_quality_score"] - ta["mean_quality_score"]),
        },
        "diversity_vs_quality": {
            "theme_tokens_effective_diff": float(
                dv["theme_tokens_effective"] - q["theme_tokens_effective"]
            ),
            "themes_covered_diff": int(dv["themes_covered"] - q["themes_covered"]),
            "min_theme_tokens_diff": int(dv["smallest_theme_tokens"] - q["smallest_theme_tokens"]),
        },
    }

    # Quality-exclusive vs random-exclusive docs isolates what the filter adds.
    q_ids, r_ids = set(frames["quality"]["id"]), set(frames["random"]["id"])
    q_only = frames["quality"][frames["quality"]["id"].isin(q_ids - r_ids)]
    r_only = frames["random"][frames["random"]["id"].isin(r_ids - q_ids)]
    comparisons["quality_vs_random"]["quality_only_docs"] = len(q_only)
    comparisons["quality_vs_random"]["random_only_docs"] = len(r_only)
    comparisons["quality_vs_random"]["quality_only_mean_score"] = float(q_only["quality_score"].mean())
    comparisons["quality_vs_random"]["random_only_mean_score"] = float(r_only["quality_score"].mean())

    payload = {
        "missing_artifacts": missing,
        "definitions": {
            "group": "head >= 10000 train docs per primary category, mid >= 1000, tail < 1000",
            "quality": "graded product of penalties in src/quality.py; 1.0 = no heuristic issue",
            "theme": "field `cluster` (0-99), hierarchical k-means on TF-IDF/SVD, fit on train only",
            "tokens": "tokenizer ids plus one EOT per document (export_tokens.py)",
            "jaccard": "|A intersect B| / |A union B| over document ids",
        },
        "inputs": {
            "processed_train": str(config.PROCESSED_DIR / "train.parquet"),
            "topics": str(config.SELECTION_DIR / "topics.parquet"),
            "manifests": str(config.MANIFEST_DIR),
            "token_files": str(config.SUBSETS_DIR),
            "tokenizer": str(config.TOKENIZER_DIR / "tokenizer.json"),
        },
        "verification": verification,
        "corpus": {
            "docs": len(full),
            "tokens": total_tokens,
            "text_tokens": text_tokens,
            "eot_tokens": len(full),
            "mean_tokens_per_doc": float(full["tokens"].mean()),
            "mean_quality_score": float(full["quality_score"].mean()),
            "exact_duplicate_text_docs": int(full["text"].duplicated().sum()),
            "group_docs": {
                group: int((full["group"] == group).sum()) for group in GROUPS
            },
            "group_tokens": {
                group: int(full.loc[full["group"] == group, "tokens"].sum())
                for group in GROUPS
            },
        },
        "comparisons": comparisons,
    }

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "selection_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary.to_csv(reports / "selection_audit_summary.csv", index=False)
    groups_out.to_csv(reports / "selection_audit_groups.csv", index=False)
    categories_out.to_csv(reports / "selection_audit_categories.csv", index=False)
    themes_out.to_csv(reports / "selection_audit_themes.csv", index=False)
    lengths_out.to_csv(reports / "selection_audit_lengths.csv", index=False)
    quality_out.to_csv(reports / "selection_audit_quality.csv", index=False)
    quality_group_out.to_csv(reports / "selection_audit_quality_by_group.csv", index=False)
    overlap.to_csv(reports / "selection_audit_overlap.csv", index=False)

    print("\nSTRATEGY SUMMARY")
    print(
        summary[
            [
                "strategy", "docs", "tokens", "mean_quality_score", "pct_low_quality",
                "group_tail_doc_share", "group_tail_token_share", "themes_covered",
                "theme_tokens_effective", "smallest_theme_tokens", "exact_duplicate_text_docs",
            ]
        ].to_string(index=False)
    )
    print("\nOVERLAP")
    print(
        overlap[overlap["strategy_a"] < overlap["strategy_b"]][
            ["strategy_a", "strategy_b", "intersection_docs", "union_docs", "jaccard_docs"]
        ].to_string(index=False)
    )
    print("\nMISSING ARTIFACTS:", missing)
    print("wrote reports/selection_audit*.{json,csv}")


if __name__ == "__main__":
    main()
