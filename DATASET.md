# DATASET.md: Part 1 Corpus Report

**Project:** Tail-Aware Data Selection for Compute-Efficient Pretraining of Compact Scientific Language Models
**Stage:** Part 1, corpus collection and analysis (complete, pending items listed in section 10)
**Report date:** 2026-10-25

---

## 1. Summary

| Item | Value |
|---|---|
| Source | `librarian-bots/arxiv-metadata-snapshot` (Hugging Face) |
| Pinned revision | `d83cc98f9ceba7b7d21cde0245233d48b63a7dbe` |
| Retrieved | 2026-10-25 13:49 UTC |
| Records scanned | 3,215,335 |
| Records matching category filter | 489,995 |
| Records after cleaning | **489,676** |
| Train / Val / Test | 440,584 / 24,401 / 24,691 |
| Unit of a document | One arXiv paper: title + abstract |
| Split method | Deterministic hash of paper ID (seed 42), 90 / 5 / 5 |

The corpus is a cs / ML subset of arXiv abstracts, with the category metadata kept so we can define head and tail topics for the selection experiments.

---

## 2. Why this source (and not the one in the synopsis)

The synopsis names `common-pile/arxiv_abstracts`. On inspection, its fields are `id`, `text`, `source`, `created`, `added` and a `metadata` dict (license, authors, submitter, URL). It has **no category field**, and it covers all of arXiv (physics, math, etc.). Without categories we cannot restrict to NLP / ML or define long-tail topics, so we switched to a metadata snapshot that includes `categories`.

The snapshot we use is derived from the Cornell University arXiv metadata dump (Kaggle `Cornell-University/arxiv`). Its HF dataset page lists the license as CC0-1.0. arXiv metadata is distributed under CC0, but the abstracts describe papers that carry their own licenses (see section 9).

---

## 3. Collection procedure

Run `uv run python build_corpus.py` from the repository root to collect,
inspect, clean, and split the corpus. To run collection by itself, use
`uv run python src/build_raw.py`.

1. Resolve and **pin** the dataset revision (written to `data/raw/source_info.json`; later runs reuse it).
2. Stream the snapshot (no full download) and keep any record whose `categories` string contains at least one of: `cs.CL`, `cs.LG`, `cs.AI`, `stat.ML`.
3. Normalize whitespace in title and abstract (removes the leading spaces and hard line breaks in the source).
4. Parse `versions` and keep the first-version timestamp as `v1_created`.
5. Save to `data/raw/arxiv_cs_ml.parquet`.

The snapshot is refreshed regularly. **Always use the pinned revision**, or the corpus will change under us.

Selection rule: a paper is included if **any** of its categories is in the target set, so many included papers have a primary category outside it (for example `cs.CV`, `cs.RO`). This is intentional, since it widens the topic tail. The primary category is the **first** category listed.

---

## 4. Raw corpus statistics (489,995 records)

### Primary category (top 25)

| Category | Papers | Category | Papers |
|---|---:|---|---:|
| cs.LG | 148,893 | eess.SP | 4,440 |
| cs.CL | 90,113 | eess.AS | 3,984 |
| cs.AI | 49,039 | cs.NE | 3,664 |
| cs.CV | 48,492 | stat.ME | 3,436 |
| stat.ML | 22,067 | quant-ph | 2,902 |
| cs.RO | 11,186 | cs.SI | 2,795 |
| cs.CR | 10,193 | cs.DC | 2,746 |
| cs.IR | 7,833 | eess.SY | 2,598 |
| eess.IV | 6,544 | math.ST | 2,262 |
| cs.SD | 6,100 | cs.IT | 2,186 |
| cs.SE | 5,690 | cs.NI | 2,099 |
| cs.CY | 5,246 | | |
| cs.HC | 5,025 | | |
| math.OC | 4,571 | | |

Membership counts for the four target categories (a paper can count in several; measured in the first exploration pass over the same 489,995 papers): cs.LG 293,609, cs.AI 207,400, cs.CL 122,720, stat.ML 81,114. The sum (about 705k) exceeds 489,995 because roughly 215k papers are cross-listed.

**Takeaway:** the corpus is ML-heavy rather than NLP-only. cs.CL is the second-largest primary category (about 18%), and cs.LG is the largest (about 30%). There is a real head and a long tail of small categories.

### Papers per year (first-version date)

| Period | Papers |
|---|---:|
| 1993 to 2010 | 5,553 |
| 2011 to 2015 | 15,991 |
| 2016 to 2019 | 58,432 |
| 2020 | 34,397 |
| 2021 | 37,827 |
| 2022 | 41,265 |
| 2023 | 51,778 |
| 2024 | 68,032 |
| 2025 | 85,148 |
| 2026 (partial) | 91,808 |

About **60%** of the corpus (296,766 papers) was first submitted in 2023 or later. The model will mostly learn recent, LLM-era scientific writing.

### Abstract length (words)

| Percentile | Words |
|---|---:|
| 0.1% | 31 |
| 1% | 65 |
| 5% | 95 |
| 50% | 171 |
| 95% | 253 |
| 99% | 279 |
| 99.9% | 303 |

arXiv caps abstract length, so the distribution has a hard upper tail. Missing or empty abstracts: 0. Missing v1 dates: 0.

---

## 5. Cleaning

Script: `src/clean_corpus.py`. Every removal is logged by reason in `data/interim/removed.parquet`, and counts are in `reports/cleaning_report.json`.

Rules, applied in this order:

| Order | Rule | Threshold | Removed |
|---:|---|---|---:|
| 1 | Empty abstract | 0 words | 0 |
| 2 | Too short | under 20 words | 146 |
| 3 | Too long | over 800 words | 0 |
| 4 | Withdrawn notice | regex: "this/the paper/article/... has been / is / was withdrawn" | 7 |
| 5 | Exact duplicate abstract | normalized (lowercase, alphanumerics only, MD5); keep lowest ID | 166 |
| | **Total removed** | | **319** |

Input 489,995, output **489,676** (0.07% removed).

Design decisions:

- **Cleaning is deliberately light.** Quality-based selection is one of our experimental conditions, so we do not remove "low-quality" documents here. We only remove records that are clearly unusable.
- The word "withdrawn" appears in 69 abstracts, but the strict notice pattern matched only 7. The rest are presumably ordinary uses of the word. A spot-check is recommended (section 10).
- `MAX_WORDS = 800` never triggers (the longest 0.1% are about 303 words). It is a safety bound only.

Training text format (fixed for all experiments):

```
{title}\n\n{abstract}
```

Controlled by `INCLUDE_TITLE = True` in `src/config.py`. Changing it changes token counts, so it must be identical across all selection strategies.

---

## 6. Splits

Script: `src/split_corpus.py`

- Assignment: `md5("42:" + paper_id) mod 10000`. Values below 9000 go to train, 9000 to 9499 to val, 9500 and above to test.
- Deterministic and stateless: the same paper ID always lands in the same split. No shuffling state to lose, and each paper appears in exactly one split.

| Split | Papers | Approx. words | Primary categories present |
|---|---:|---:|---:|
| Train | 440,584 | 80,285,726 | 152 |
| Val | 24,401 | 4,427,682 | 126 |
| Test | 24,691 | 4,499,275 | 129 |

Category proportions match across splits to within a fraction of a percentage point (for example cs.LG: 30.3% train, 30.9% val, 31.0% test; cs.CL: 18.4%, 18.2%, 18.1%).

**Test-set coverage caveat:** only 26 categories have at least 100 test documents, and 85 have fewer than 30. Per-category perplexity will be very noisy for tail categories. Evaluate by **group** (head / mid / tail) instead of by individual category (section 10).

Usage rules:

- Train pool: candidate documents for every selection strategy. Tokenizer is fit on **train only**.
- Validation: monitoring and development decisions.
- Test: final evaluation only. Never used for scoring, selection, or tuning.

---

## 7. Files and schema

```
data/
  raw/
    arxiv_cs_ml.parquet      352 MB   filtered snapshot, uncleaned
    source_info.json                  pinned revision + retrieval metadata
  interim/
    cleaned.parquet          693 MB   after cleaning, before split
    removed.parquet          4.7 KB   id + reason for each removed record
  processed/
    train.parquet            624 MB
    val.parquet               35 MB
    test.parquet              36 MB
    split_manifest.parquet   2.9 MB   id, primary_category, split
reports/
  raw_stats.json  cleaning_report.json  split_report.json
  figures/  primary_category_top30.png  category_long_tail.png
            papers_per_year.png  abstract_length.png
```

`data/` is git-ignored. Share the pinned revision and scripts, not the parquet files.

Columns in `train/val/test.parquet`:

| Column | Type | Notes |
|---|---|---|
| `id` | str | arXiv ID; the stable document ID and the split key |
| `title` | str | whitespace-normalized |
| `abstract` | str | whitespace-normalized; may contain LaTeX / math |
| `categories` | list[str] | all categories, space-split from the source |
| `primary_category` | str | first listed category |
| `v1_created` | datetime (UTC) | first-version date; **use this for any time analysis** |
| `n_versions` | int | number of versions |
| `update_date` | str | date of last revision, **not** publication date |
| `journal_ref` | str / null | |
| `license` | str / null | per-paper license URL from arXiv |
| `n_words` | int | abstract word count |
| `text` | str | training text: `title\n\nabstract` |
| `split` | str | `train` / `val` / `test` |

---

## 8. Reproducing the corpus

From the project root, with `uv` installed:

```bash
uv sync
uv run python src/build_raw.py        # about 12 min; reuses the pinned revision if source_info.json exists
uv run python src/inspect_corpus.py   # statistics + figures
uv run python src/clean_corpus.py     # logged cleaning
uv run python src/split_corpus.py     # 90/5/5 hash split
```

All settings (categories, seed, fractions, thresholds, title inclusion) live in `src/config.py`. To rebuild exactly this corpus, keep `data/raw/source_info.json` with revision `d83cc98f9ceba7b7d21cde0245233d48b63a7dbe`.

A Hugging Face token is optional (it only speeds up streaming). If you use one, store it in an untracked `.env`.

---

## 9. Known limitations and risks

1. **Near-duplicates remain.** Only exact duplicate abstracts were removed. Extended versions of a paper under a different ID, or lightly edited abstracts, can fall into different splits and leak between train and test. Likely a small effect; not yet measured. A MinHash near-dedup pass is a possible extension.
2. **Thin tail in the test set.** See section 6. Evaluate by topic group, not by individual category.
3. **Temporal skew.** About 60% of papers date from 2023 or later. Results describe recent scientific writing and may not transfer to older text.
4. **Not NLP-only.** The corpus spans ML, vision, robotics, security, signal processing and more. This is by design for tail analysis, but results should not be described as "NLP-specific".
5. **Abstracts only.** No section structure (introduction, related work, conclusion). Section-level data would require full-text sources and is out of scope for the core project.
6. **Math and LaTeX in text.** Abstracts contain `$...$` expressions. Quality heuristics based on symbol ratios must be calibrated so math-heavy abstracts are not unfairly penalized.
7. **Cross-listing.** About 215k papers carry several target categories. We use the primary (first) category for topic labels; this is a modelling choice and should be stated in any results.
8. **Licensing.** The snapshot is CC0 at the metadata level, but abstracts describe papers with their own licenses, recorded per paper in the `license` column. Maintainers of related datasets note that license metadata can be inaccurate. For a course project this is low risk; do not redistribute the corpus without checking terms.
9. **Snapshot drift.** The source is refreshed periodically. Only the pinned revision is reproducible.

---

## 10. Pending items (not yet measured)

| Item | Why it matters | How to get it |
|---|---|---|
| Head / mid / tail group sizes | Needed for group-wise evaluation. Proposed definition: head = at least 10,000 train papers per primary category, mid = 1,000 to 9,999, tail = under 1,000. This is a proposal for the team to confirm. | Run the group-count snippet below |
| Token counts of the corpus | Needed to set the training-token budget (target: about 20 to 40% of the pool) | After the BPE tokenizer is trained in Part 2 |
| "Withdrawn" spot-check | 62 abstracts mention the word but were not removed | Print a sample and read it |
| Near-duplicate rate | Quantifies limitation 1 | Optional MinHash pass |

Group-count snippet:

```bash
uv run python - <<'EOF'
import pandas as pd
tr = pd.read_parquet("data/processed/train.parquet", columns=["primary_category"])
te = pd.read_parquet("data/processed/test.parquet", columns=["primary_category"])
c = tr["primary_category"].value_counts()
grp = lambda n: "head" if n >= 10_000 else ("mid" if n >= 1_000 else "tail")
g = c.map(grp)
print("categories per group:\n", g.value_counts().to_string())
print("\ntrain docs per group:\n", tr["primary_category"].map(g).value_counts().to_string())
print("\ntest docs per group:\n", te["primary_category"].map(g).value_counts(dropna=False).to_string())
EOF
```

---

## 11. Part 1 checklist

- [x] Corpus source and schema documented
- [x] Raw sample inspected and metadata fields verified
- [x] Cleaning rules implemented and logged
- [x] Duplicate handling defined (exact only; near-dups listed as a limitation)
- [x] Topic distribution and corpus statistics plotted
- [x] Train, validation and test splits created
- [x] Dataset version, seeds and preprocessing settings recorded

Next: Part 2, data selection and tokenization (quality scoring, head / mid / tail definition, BPE tokenizer trained on train only).
