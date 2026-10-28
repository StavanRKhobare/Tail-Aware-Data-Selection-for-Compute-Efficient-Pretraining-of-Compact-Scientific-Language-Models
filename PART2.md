# PART2.md: Data Selection and Tokenization

**Project:** Tail-Aware Data Selection for Compute-Efficient Pretraining of Compact Scientific Language Models
**Stage:** Part 2, data selection and tokenization (complete; decisions for the team in section 10)
**Input:** Part 1 splits (`data/processed/*.parquet`, also on Kaggle `vroomvroom2006/processedryan`)

---

## 1. Summary

| Item | Value |
|---|---|
| Tokenizer | Byte-level BPE, vocab 16,384, trained on the train split only |
| Train pool | 440,584 docs, 108.5M tokens (incl. one end-of-text token per doc) |
| Token budget per subset | **30,000,000** (27.6% of the pool) |
| Subsets | `random`, `quality`, `diversity`, `tail_aware` (proposed), `tail_only` (ablation) |
| Leakage removed | 122 train docs that are versions of a val / test paper |
| Near-duplicates found | 455 train docs (revised / extended versions of another train paper) |
| Quality filter removes | 2.6% of train (score < 0.5) + near-duplicates |
| Tail share of tokens | random 4.5%, tail_aware **16.1%** (3.5x) |
| Hand-off to Part 3 | `data/subsets/<name>/train.bin`, `val.bin`, `test.bin` (uint16 token ids) |

**The two most important findings so far** (before any model is trained):

1. **arXiv abstracts are already clean.** 88.2% of abstracts trigger no quality heuristic at all; only 2.6% have a clear defect. Heuristic quality filtering can change little here, so `quality` is expected to perform close to `random`. This is a property of the corpus, not a bug.
2. **Content diversity and field coverage are different axes.** Balancing text clusters reshapes *what the papers talk about* (theme shift 0.158) but leaves the *field* mix almost untouched (tail stays ~4.8%). Rare fields (for example ML for chemistry) use mostly ML vocabulary, so text clusters file them next to ordinary ML papers. Covering the field tail needs the field labels, which is what `tail_aware` uses.

---

## 2. Run order and runtime

```bash
uv sync
uv run python src/train_tokenizer.py   # ~6 min   tokenizer + token counts
uv run python src/score_quality.py     # ~4 min   quality, near-duplicates, leakage
uv run python src/build_topics.py      # ~7 min   head / mid / tail + text themes
uv run python src/select_subsets.py    # ~10 s    the 5 subsets
uv run python src/export_tokens.py     # ~3 min   .bin files for Part 3
uv run pytest                          # 25 unit tests, ~2 s
```

Timings on 4 CPU cores, no GPU. All settings live in `src/config.py` (Part 2 section).

---

## 3. Tokenizer

Script: `src/train_tokenizer.py`. Output: `tokenizer/tokenizer.json` (**committed**; this is the fixed tokenizer for every experiment).

- **Type:** byte-level BPE (GPT-2 style). Any text, including LaTeX and accents, can be encoded; there is no "unknown" token. NFC unicode normalization.
- **Trained on:** all 440,584 train documents (`title\n\nabstract`). Val / test are never seen. The tokenizer is shared by all strategies, so it cannot favour one of them.
- **Special tokens:** `<|endoftext|>` = id 0 (appended after every document), `<|pad|>` = id 1.
- **Round trip is exact:** `decode(encode(text)) == text` (tested).

**Choosing the vocabulary size** (trained on a 100k-doc train sample, measured on val):

| Vocab | Tokens per word (val) | Embedding params at d=384 |
|---:|---:|---:|
| 8,192 | 1.464 | 3.1M |
| **16,384** | **1.347** | **6.3M** |
| 32,768 | 1.295 | 12.6M |

Doubling 16k to 32k saves only 4% of tokens but doubles the embedding table, which would consume most of a 10 to 20M-parameter budget. 16k is the balance point.

**Tokenizer fairness across topics** (val, tokens per word): head 1.345, mid 1.351, tail 1.371. Tail text costs about 2% more tokens. Small, but it means per-token perplexities of different groups are not perfectly comparable; compare *strategies within a group*, not groups with each other.

Example: `We prove an $O(\sqrt{T}\log T)$ regret bound` becomes `We | Ġprove | Ġan | Ġ$ | O | (\ | sqrt | { | T | }\ | log | ĠT | )$ | Ġregret | Ġbound` (`Ġ` marks a preceding space).

Corpus token counts: train 108.1M, val 5.96M, test 6.07M (document tokens, before end-of-text tokens).

---

## 4. Quality scoring

Scripts: `src/quality.py` (signals), `src/score_quality.py`. Output: `data/selection/quality.parquet`, `reports/quality_report.json`.

### 4.1 Signals

All signals are computed on the abstract. **LaTeX math (`$...$`, `\(...\)`, `\[...\]`) and URLs are removed before the symbol and language checks**, so math-heavy papers are not penalized for their math (DATASET.md limitation 6). Math share is reported separately.

Thresholds were chosen by printing abstracts at the extremes of each signal and reading them. Each rule gives a penalty that is 0 inside the normal range and rises linearly to 1:

| Signal | Penalty starts | Full penalty | What it catches | Docs with any penalty |
|---|---:|---:|---|---:|
| abstract words (low) | 80 | 40 | near-empty abstracts | 2.2% |
| chars in repeated word 3-grams | 0.15 | 0.35 | repetitive phrasing | 3.6% |
| top word 2-gram char share | 0.10 | 0.20 | one phrase dominates | 0.6% |
| non-prose symbols outside math | 0.005 | 0.03 | `**markdown**`, stray markup | 2.8% |
| LaTeX commands outside math | 0 | 6 | `\citep{...}`, `\textit{}` | 4.5% |
| mean sentence length (high) | 40 words | 80 | run-on text, missing punctuation | 0.7% |
| mean sentence length (low) | 10 words | 5 | fragments, lists | <0.01% |
| stopword share (low) | 0.20 | 0.10 | not running prose | <0.1% |
| foreign function words | 0.02 | 0.06 | non-English text | <0.1% |
| ALL-CAPS word share | 0.12 | 0.25 | acronym soup | 0.1% |

`quality_score = product of (1 - penalty)` over all rules: 1.0 = no issue, 0 = severe issue. The filter keeps documents with **score >= 0.5** (`QUALITY_MIN`) that are not near-duplicates.

Signals that were measured and dropped: the non-ASCII share is 0 for essentially every abstract (arXiv writes accents as LaTeX, e.g. `\"o`). A low stopword share turned out to mean dense technical prose, not junk, so its threshold is very low. The foreign-word check only counts lowercase words, because `LOS` (length of stay) and `DAS` look like Spanish / German function words.

### 4.2 Results

| | Value |
|---|---:|
| Documents with no issue (score 1.0) | 88.2% |
| Score below 0.5 | 2.6% |
| 1st / 5th / 10th percentile score | 0.15 / 0.73 / 0.93 |

**Does the filter hurt the tail?** Slightly. It removes 3.8% of tail documents vs 2.6% of head and 2.7% of mid (`reports/figures/quality_removed_by_group.png`). This is the tension the project studies: quality filters can quietly shrink rare topics.

---

## 5. Near-duplicates and train / test leakage

Script: `src/minhash.py` (vectorized MinHash + LSH, numpy only), used by `src/score_quality.py`.

- Each abstract is turned into its set of word 5-grams; 128 MinHash values estimate the Jaccard similarity of any two sets. LSH with 32 bands x 4 rows proposes candidate pairs (catch probability about 87% at similarity 0.5), which are then verified on the full signature.
- **Threshold 0.5.** Pairs were read by hand at several similarity levels. At 0.5 to 0.8 they are revised or extended versions of the same paper under a different arXiv id (e.g. "the $\ell^1$-norm" vs "the l1-norm", conference vs journal version). Below 0.5 they are mostly shared templates ("This is the Proceedings of the ... Conference on Uncertainty in AI"). A first run at 0.8 missed most of the revised versions.
- **Within train:** 472 verified pairs, 455 documents marked `near_dup` (each cluster keeps its lowest id, as in Part 1). These are removed by the quality filter.
- **Leakage:** a train document is `contaminated` if it is similar (>= 0.5) to a val / test abstract **or** has an identical normalized title. 122 train documents (91 by abstract, 43 by title, some both). They are excluded from **every** subset, including `random`.

Val / test text is used only for this leakage check; no val / test statistic influences any score, threshold or selection rule. This is standard decontamination and is applied identically to all strategies.

---

## 6. Topics

Script: `src/build_topics.py`, `src/topics.py`. Output: `data/selection/topics.parquet` (train, val and test), `reports/topics_report.json`.

### 6.1 Head / mid / tail groups (from arXiv field labels)

Defined by the number of **train** papers per primary category (the definition proposed in DATASET.md section 10):

| Group | Rule | Categories | Train docs | Val docs | Test docs |
|---|---|---:|---:|---:|---:|
| head | >= 10,000 | 6 | 332,367 (75.4%) | 18,575 | 18,592 |
| mid | 1,000 to 9,999 | 28 | 88,242 (20.0%) | 4,752 | 4,954 |
| tail | < 1,000 | 118 | 19,975 (4.5%) | 1,074 | 1,145 |

Head: cs.AI, cs.CL, cs.CV, cs.LG, cs.RO, stat.ML. A category that never occurs in train is treated as tail.

With ~1,100 test documents (~280k tokens), the tail **group** is large enough for a stable perplexity estimate even though individual tail categories are not.

### 6.2 Text themes (from the words in the abstracts)

TF-IDF (50k words, math removed) -> SVD to 100 dimensions -> **hierarchical k-means** (Vo et al., 2024): 1,000 fine clusters, whose centroids are then clustered into 100 themes. Fit on train only; val / test are assigned to the train clusters.

Why hierarchical: flat k-means (k = 100) was tried first. It places more clusters where papers are dense, so its clusters came out nearly equal in size (1.9k to 9.2k docs), and balancing them changed the data very little (theme shift 0.11, 79% document overlap with `quality`). In the hierarchical version every fine cluster counts once in the second level, which flattens that density: themes range from 478 to 11,055 docs, and balancing them gives a theme shift of 0.158.

Largest themes: "AI / human / society" (11.1k docs), "graph neural networks" (10.9k), "LLM agents" (10.1k), "reinforcement learning" (9.9k). Smallest: "speech translation / ASR" (478), "online learning / regret" (506), "federated learning attacks" (564).

Themes cut across fields: a head category spreads over a median of 99.5 of the 100 themes, a tail category over 24.5.

---

## 7. Selection strategies

Scripts: `src/sampling.py` (mechanism), `src/select_subsets.py` (strategies). Outputs: `manifests/<name>.txt.gz` (**committed**: sorted ids of every subset), `data/selection/subsets/<name>.parquet`, `reports/selection_report.json`.

### 7.1 Design

A small factorial design, so each comparison isolates one ingredient:

| | Natural mix | Balanced over text themes | Tail-aware over fields |
|---|---|---|---|
| **No quality filter** | `random` | | `tail_only` (ablation) |
| **Quality filter** | `quality` | `diversity` | **`tail_aware` (proposed)** |

| Comparison | Isolates |
|---|---|
| `quality` vs `random` | the quality filter |
| `tail_aware` vs `quality` | tail-aware field sampling (on top of quality) |
| `diversity` vs `quality` | content-theme balancing |
| `tail_aware` vs `diversity` | field-tail coverage vs content diversity |
| `tail_aware` vs `tail_only` | the quality filter when the tail is upweighted |

### 7.2 Mechanism (identical for all strategies)

1. **Candidate pool.** `eligible` = train minus the 122 leaked documents. `quality` = eligible minus score < 0.5 minus near-duplicates.
2. **Common random numbers.** Every train document gets one random key (seed 42). Every strategy takes documents in that same key order. So `quality` is `random` except for the filtered documents and their replacements (95% of documents shared), and `tail_only` vs `tail_aware` likewise (95% shared). Differences between subsets come from the rules, not from sampling luck.
3. **Topic allocation** (topic-aware strategies only). The budget is split across topics with weight `tokens_c ** alpha`. A topic never gets more than it has; what it cannot use is shared among the others ("water-filling").
   - `alpha = 1`: the natural mix. `alpha = 0`: every topic gets the same amount. In between: rare topics are boosted, but not to full equality. This is the temperature sampling used for multilingual LMs (e.g. XLM-R, mT5) to boost low-resource languages.
   - `tail_aware` / `tail_only`: topic = primary category, `alpha = 0.5`.
   - `diversity`: topic = text theme, `alpha = 0` (every theme equal).
4. **Fill.** Documents are taken in key order until the (per-topic) budget is used; a small remainder is topped up in key order. Every subset ends within one document of 30M tokens.

### 7.3 What each subset contains (30M tokens each)

| | random | quality | diversity | tail_aware | tail_only |
|---|---:|---:|---:|---:|---:|
| Documents | 121,766 | 121,486 | 122,526 | 120,408 | 120,859 |
| Head / mid / tail tokens | 74.9 / 20.6 / 4.5% | 74.9 / 20.6 / 4.5% | 76.2 / 19.0 / 4.8% | **38.9 / 45.0 / 16.1%** | 38.6 / 44.8 / 16.5% |
| Tail upweighting vs train | 0.99x | 0.99x | 1.05x | **3.51x** | 3.62x |
| Effective number of fields | 14.2 | 14.1 | 13.6 | **45.4** | 46.1 |
| Fields with >= 50 docs | 71 | 69 | 68 | 106 | 107 |
| Tail fields covered (of 118) | 106 | 106 | 111 | 117 | 118 |
| Effective number of themes | 90.1 | 90.0 | **99.3** | 84.1 | 84.2 |
| Field-mix shift vs train | 0.007 | 0.007 | 0.037 | **0.366** | 0.369 |
| Theme-mix shift vs train | 0.010 | 0.011 | **0.158** | 0.129 | 0.130 |
| Docs with score < 0.5 | 2.7% | 0 | 0 | 0 | 2.8% |
| Near-duplicates | 131 | 0 | 0 | 0 | 129 |
| Docs from 2023 or later | 60.6% | 60.9% | 58.6% | 58.6% | 58.2% |

*Effective number* = exp(entropy) of the token mix: how many equally-sized topics the mix is worth. *Shift* = total variation distance from the full train mix (0 = same mix, 1 = no overlap).

Document overlap (Jaccard): random-quality 0.95, tail_aware-tail_only 0.95, quality-diversity 0.73, quality-tail_aware 0.46.

Figures: `reports/figures/selection_group_mix.png`, `selection_category_profile.png`.

Side effects to keep in mind when reading Part 4 results:

- `tail_aware` halves the head (75% to 39% of tokens). The test set is 75% head, so its **overall** perplexity may get worse even if tail perplexity improves. Report by group.
- The tail-aware and diversity subsets are slightly older (58.6% vs 60.6% of docs from 2023 or later), because some rare fields and themes have older papers. A small confound; the year of every selected doc is in `data/selection/subsets/<name>.parquet`.

---

## 8. Files for Part 3 and Part 4

Script: `src/export_tokens.py`. Everything in `data/subsets/` is git-ignored (rebuild with the script):

```
data/subsets/
  random/train.bin  quality/train.bin  diversity/train.bin
  tail_aware/train.bin  tail_only/train.bin     60 MB each: 30.0M tokens, documents shuffled (seed 42)
  val.bin  test.bin                             full val / test, documents in id order
  val_docs.parquet  test_docs.parquet           id, start, end (token offsets), primary_category,
                                                group, fine_cluster, cluster
  meta.json                                     vocab_size 16384, eot_id 0, pad_id 1, dtype uint16
```

Format (nanoGPT-style): one flat array of uint16 token ids; each document is followed by `<|endoftext|>` (id 0).

```python
import numpy as np
data = np.memmap("data/subsets/tail_aware/train.bin", dtype=np.uint16, mode="r")
# sample a training batch: random windows of length block_size + 1
ix = np.random.randint(0, len(data) - 257, size=32)
x = np.stack([data[i : i + 256] for i in ix]); y = np.stack([data[i + 1 : i + 257] for i in ix])

# group-wise evaluation in Part 4: per-document token ranges of val / test
import pandas as pd
docs = pd.read_parquet("data/subsets/val_docs.parquet")   # start / end index into val.bin
```

Recommendations for Part 3:

1. Train every subset for the **same number of tokens** (same steps x batch x context). The subsets all hold 30.0M tokens, so the same number of epochs is the same compute.
2. Keep model, optimizer, schedule and seed identical across subsets. If time allows, 2 to 3 training seeds per subset; differences between conditions may be small.
3. If GPU time is short, train in this order: `random`, `tail_aware`, `quality`, `diversity`, `tail_only`.
4. Use `val.bin` for monitoring and early decisions; keep `test.bin` for the final numbers only.

---

## 9. Reproducibility

- `tokenizer/tokenizer.json` and `manifests/*.txt.gz` are committed: they *are* the experiment inputs. Anyone can rebuild the `.bin` files from them plus the Part 1 parquet files (`src/export_tokens.py` reads the manifests).
- Re-running the full pipeline reproduces the manifests given the same library versions. Clustering (k-means) can differ slightly across scikit-learn versions; if it does, trust the committed manifests.
- Replicate subsets with a different random key: `uv run python src/select_subsets.py --seed 7` (writes `*_seed7` manifests and report). Other budgets: `--budget 20000000`.

---

## 10. Decisions for the team to confirm

| Decision | Current value | Why | Alternative worth considering |
|---|---|---|---|
| Token budget | 30M per subset (27.6% of pool) | Within the planned 20 to 40%; leaves room for selection to matter | 20M if training is slow |
| Tail temperature | alpha = 0.5 | Standard middle ground; tail 3.5x, head halved | An alpha sweep (0.3 / 0.7) in Part 4 shows the trade-off curve |
| Quality threshold | 0.5 | Removes only clear defects (2.6%) | 0.8 removes 5.7%; still small either way |
| Group thresholds | 10,000 / 1,000 train docs | As proposed in DATASET.md | |
| Leakage threshold | Jaccard 0.5 or same title | Hand-checked pairs | |

---

## 11. Limitations

1. **Weak quality signal.** Heuristics find little to remove in arXiv abstracts; `quality` vs `random` will likely show a small or null effect. A model-based quality classifier (as in DCLM or FineWeb-Edu) is a possible extension but needs labelled "good" examples.
2. **tail_aware changes two things at once:** more tail and less head. Per-group evaluation separates the two effects; an alpha sweep would map the trade-off.
3. **Field labels are the primary category only** (cross-listing ignored), as in Part 1.
4. **Themes are lexical.** TF-IDF + SVD (10.7% of variance kept) captures shared vocabulary, not meaning. Neural sentence embeddings would give better themes at higher cost.
5. **One random key (seed 42).** Use `--seed` for replicates if results are close.
6. **Small temporal shift** in tail_aware / diversity (section 7.3).
7. **Leakage check is lexical.** A paper rewritten entirely under a new id and title is not detected.

---

## 12. Part 2 checklist

- [x] BPE tokenizer trained on train only, vocab size justified, committed
- [x] Token counts for every document; budget set (30M tokens)
- [x] Quality signals (math-aware), thresholds calibrated by reading examples
- [x] Near-duplicates (MinHash) and train / test leakage removed
- [x] Head / mid / tail groups and text themes for train, val and test
- [x] Five subsets at an identical token budget, with coverage statistics and figures
- [x] Tokenized `.bin` files and per-document offsets for Parts 3 and 4
- [x] Unit tests (`uv run pytest`)

Next: Part 3, train the same compact Transformer on each subset.
