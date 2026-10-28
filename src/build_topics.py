"""Topic labels for every document: head / mid / tail group and a text cluster.

Groups come from primary-category counts in train. Clusters are hierarchical
k-means (Vo et al., 2024) on TF-IDF -> SVD (100 dims) embeddings, fit on train
only: 1,000 fine clusters, then their centroids are clustered into 100 themes.
Flat k-means puts more clusters where documents are dense, so its clusters
come out nearly equal in size and balancing over them changes little; each
fine centroid counts once in the second level, which flattens that density.
Val / test documents are only assigned to the train clusters (Part 4).

Outputs
  data/selection/topics.parquet   id, split, primary_category, group,
                                  fine_cluster (0-999), cluster (theme, 0-99)
  reports/topics_report.json      group sizes, cluster sizes and top terms
  reports/figures/topic_*.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config
from quality import MATH_RE
from topics import GROUPS, assign_groups, category_groups

SVD_DIMS = 100


def strip_math(texts):
    return [MATH_RE.sub(" ", t) for t in texts]


def main():
    config.ensure_dirs()
    cols = ["id", "text", "primary_category"]
    splits = {s: pd.read_parquet(config.PROCESSED_DIR / f"{s}.parquet", columns=cols) for s in ("train", "val", "test")}
    train = splits["train"]
    report = {}

    # 1. Head / mid / tail groups (train counts define the groups for every split).
    cat_to_group = category_groups(train["primary_category"])
    for name, df in splits.items():
        df["group"] = assign_groups(df["primary_category"], cat_to_group)
    report["groups"] = {
        "definition": f"head >= {config.HEAD_MIN_DOCS:,} train docs per primary category, "
                      f"mid >= {config.MID_MIN_DOCS:,}, tail below",
        "categories_per_group": cat_to_group.value_counts().reindex(GROUPS).to_dict(),
        "docs_per_group": {s: df["group"].value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict()
                           for s, df in splits.items()},
        "head_categories": sorted(cat_to_group[cat_to_group == "head"].index),
        "mid_categories": sorted(cat_to_group[cat_to_group == "mid"].index),
    }

    # 2. Text clusters, fit on train only.
    vec = TfidfVectorizer(max_features=50_000, min_df=5, max_df=0.5, stop_words="english",
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", dtype=np.float32)
    x_train = vec.fit_transform(strip_math(train["text"].tolist()))
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=config.SEED)
    z_train = normalize(svd.fit_transform(x_train))
    fine = KMeans(n_clusters=config.N_FINE_CLUSTERS, random_state=config.SEED, n_init=1, max_iter=100)
    train["fine_cluster"] = fine.fit_predict(z_train)
    theme = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.SEED, n_init=10)
    fine_to_theme = theme.fit_predict(normalize(fine.cluster_centers_))
    train["cluster"] = fine_to_theme[train["fine_cluster"]]
    for s in ("val", "test"):
        z = normalize(svd.transform(vec.transform(strip_math(splits[s]["text"].tolist()))))
        splits[s]["fine_cluster"] = fine.predict(z)
        splits[s]["cluster"] = fine_to_theme[splits[s]["fine_cluster"]]
    print("clustering done")

    # Describe each theme: top TF-IDF terms of its centroid and its main categories.
    terms = np.array(vec.get_feature_names_out())
    centroid_terms = svd.inverse_transform(theme.cluster_centers_)
    sizes = train["cluster"].value_counts()
    clusters = []
    for c in sizes.index:
        members = train[train["cluster"] == c]
        cats = members["primary_category"].value_counts(normalize=True).head(3)
        grp = members["group"].value_counts(normalize=True)
        clusters.append({
            "cluster": int(c),
            "train_docs": int(sizes[c]),
            "fine_clusters": int((fine_to_theme == c).sum()),
            "top_terms": terms[np.argsort(-centroid_terms[c])[:8]].tolist(),
            "top_categories": {k: round(float(v), 3) for k, v in cats.items()},
            "tail_share": round(float(grp.get("tail", 0.0)), 3),
        })
    report["clusters"] = {
        "method": f"TF-IDF (50k unigrams, math removed) -> SVD {SVD_DIMS} -> KMeans k={config.N_FINE_CLUSTERS} "
                  f"-> KMeans k={config.N_CLUSTERS} on the fine centroids (themes)",
        "train_docs_per_cluster": {"min": int(sizes.min()), "median": int(sizes.median()), "max": int(sizes.max())},
        "train_docs_per_fine_cluster": {"min": int(train["fine_cluster"].value_counts().min()),
                                        "median": int(train["fine_cluster"].value_counts().median()),
                                        "max": int(train["fine_cluster"].value_counts().max())},
        "svd_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        "clusters": clusters,
    }

    # Clusters cut across categories: how many clusters does each category spread over?
    spread = train.groupby("primary_category")["cluster"].nunique()
    report["clusters"]["clusters_per_category_median"] = {
        g: float(spread[cat_to_group[cat_to_group == g].index].median()) for g in GROUPS
    }

    out = pd.concat([df[["id", "primary_category", "group", "fine_cluster", "cluster"]].assign(split=s)
                     for s, df in splits.items()], ignore_index=True)
    out.to_parquet(config.SELECTION_DIR / "topics.parquet", index=False)
    (config.REPORTS_DIR / "topics_report.json").write_text(json.dumps(report, indent=2))

    # Figures
    counts = train["primary_category"].value_counts()
    colors = counts.index.map(cat_to_group).map({"head": "C0", "mid": "C1", "tail": "C2"})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values, color=colors, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("primary category (ranked by size)")
    ax.set_ylabel("train documents (log)")
    ax.set_title("Head / mid / tail categories (train)")
    for g, col in (("head", "C0"), ("mid", "C1"), ("tail", "C2")):
        ax.bar([], [], color=col, label=f"{g}: {int((cat_to_group == g).sum())} categories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "topic_groups.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(sizes)), sizes.values, width=1.0)
    ax.set_xlabel("theme (ranked by size)")
    ax.set_ylabel("train documents")
    ax.set_title(f"Text theme sizes ({config.N_CLUSTERS} themes from {config.N_FINE_CLUSTERS} fine clusters)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "topic_cluster_sizes.png", dpi=150)
    plt.close(fig)

    print(json.dumps(report["groups"], indent=2))
    print(json.dumps({k: v for k, v in report["clusters"].items() if k != "clusters"}, indent=2))
    for c in clusters[:5] + clusters[-5:]:
        print(c)


if __name__ == "__main__":
    main()
