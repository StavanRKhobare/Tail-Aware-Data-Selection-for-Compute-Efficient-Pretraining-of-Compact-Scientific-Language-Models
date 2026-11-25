# Dataset Selection Audit

**Date:** 2026-11-10
**Scope:** the five selected training subsets used by the ten Part 3 runs.
**Script:** `audit_selection.py` (read-only; does not modify `data/`, `manifests/`, or `runs/`).
**Outputs:** `reports/selection_audit*.json|csv`.

---

## 1. Artifact availability

Three artifacts produced by the original Part 2 selection run are **not present
in this checkout**:

- `data/selection/quality.parquet`
- `data/selection/token_counts.parquet`
- `data/selection/subsets/*.parquet`

This is reported explicitly rather than reconstructed silently. The audit uses
only canonical artifacts that *are* present:

| Artifact | Path | Use |
|---|---|---|
| train documents | `data/processed/train.parquet` | text, abstract, category |
| topic labels | `data/selection/topics.parquet` | group, fine cluster, theme |
| selected document ids | `manifests/<strategy>.txt.gz` | membership |
| tokenized subsets | `data/subsets/<strategy>/train.bin` | exact token counts |
| tokenizer | `tokenizer/tokenizer.json` | recompute lengths, quality |

Quality scores were **recomputed** from `src/quality.py` on the canonical
abstract text. They therefore reproduce the project's own definition, not a new
one, and were validated against the aggregate checks in
`reports/quality_report.json` (mean score and low-quality removal rate match the
documented behaviour; see Section 6).

Token counts were reconstructed exactly: the export writes a seeded permutation
(`np.random.default_rng(42).permutation(n)`) of the manifest ids and one EOT per
document, so the k-th EOT-delimited span in `train.bin` belongs to the k-th
permuted id. For every strategy, recomputed token lengths matched the `.bin`
spans with **0 mismatches**.

### Verification

| Strategy | Docs | Unique ids | `.bin` docs | `.bin` tokens | Length mismatches | Ends in EOT | Double-EOT pairs |
|---|---:|---:|---:|---:|---:|---|---:|
| random | 121,766 | 121,766 | 121,766 | 29,999,918 | 0 | yes | 0 |
| quality | 121,486 | 121,486 | 121,486 | 29,999,935 | 0 | yes | 0 |
| diversity | 122,526 | 122,526 | 122,526 | 29,996,698 | 0 | yes | 0 |
| tail_aware | 120,408 | 120,408 | 120,408 | 29,999,918 | 0 | yes | 0 |
| tail_only | 120,859 | 120,859 | 120,859 | 29,998,854 | 0 | yes | 0 |

No duplicate document ids and no consecutive EOTs in any subset. The token files
are internally consistent.

---

## 2. Definitions used

These are the project's existing definitions, not new ones:

- **Group (head/mid/tail):** based on train document count per primary arXiv
  category; head ≥ 10,000 docs, mid ≥ 1,000, tail < 1,000
  (`config.HEAD_MIN_DOCS`, `config.MID_MIN_DOCS`, `src/topics.py`).
- **Theme (`cluster`, 0–99):** second-level hierarchical k-means over
  TF-IDF → SVD features, fit on train only (`src/build_topics.py`).
- **Fine cluster (`fine_cluster`, 0–999):** first-level k-means, train only.
- **Quality score:** graded product of the rule penalties in `src/quality.py`;
  `1.0` means no heuristic issue, `< 0.5` fails `QUALITY_MIN`.
- **Token count:** tokenizer ids plus one EOT per document, exactly as written
  by `src/export_tokens.py`.
- **Jaccard:** `|A ∩ B| / |A ∪ B|` over document ids.

Full train corpus for reference: **440,584 documents**, **108,540,610 tokens**
(108,100,026 text tokens + 440,584 EOTs), mean 246.36 tokens/doc.
Group token shares: head 74.86%, mid 20.57%, tail 4.57%.

---

## 3. Size and composition

| Strategy | Docs | Unique | Tokens | Mean tok/doc | Doc share of train | Token share of train |
|---|---:|---:|---:|---:|---:|---:|
| full train | 440,584 | 440,584 | 108,540,610 | 246.36 | 1.000 | 1.000 |
| random | 121,766 | 121,766 | 29,999,918 | 246.37 | 0.2764 | 0.2764 |
| quality | 121,486 | 121,486 | 29,999,935 | 246.94 | 0.2757 | 0.2764 |
| diversity | 122,526 | 122,526 | 29,996,698 | 244.84 | 0.2781 | 0.2764 |
| tail_aware | 120,408 | 120,408 | 29,999,918 | 249.15 | 0.2733 | 0.2764 |
| tail_only | 120,859 | 120,859 | 29,998,854 | 248.22 | 0.2743 | 0.2764 |

All five sit at the same 30M-token budget (the Part 3 runs then make 2 epochs,
60M tokens). Document count varies by about 1.8% because token length, not
document count, is held fixed. This is the correct design for an equal-compute
comparison.

### Document length distribution

| Strategy | Min | p10 | p25 | Median | p75 | p90 | p99 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full train | 28 | 159 | 198 | 243 | 292 | 340 | 420 | 1271 | 246.4 |
| random | 30 | 159 | 198 | 243 | 292 | 340 | 420 | 1271 | 246.4 |
| quality | 73 | 161 | 199 | 243 | 292 | 339 | 418 | 718 | 246.9 |
| diversity | 73 | 158 | 197 | 241 | 289 | 337 | 417 | 718 | 244.8 |
| tail_aware | 74 | 159 | 198 | 244 | 296 | 345 | 427 | 718 | 249.2 |
| tail_only | 28 | 156 | 197 | 244 | 296 | 346 | 429 | 1271 | 248.2 |

`quality`, `diversity`, and `tail_aware` remove very short and very long
documents (min 73, max 718) because the quality filter penalises them.
`tail_only` deliberately skips the quality filter and retains the wider length
range, including a 1,271-token document.

---

## 4. Topic composition

### 4.1 Head / mid / tail documents

| Strategy | Head docs | Mid docs | Tail docs |
|---|---:|---:|---:|
| random | 91,858 (75.44%) | 24,417 (20.05%) | 5,491 (4.51%) |
| quality | 91,733 (75.51%) | 24,325 (20.02%) | 5,428 (4.47%) |
| diversity | 94,076 (76.78%) | 22,649 (18.49%) | 5,801 (4.73%) |
| tail_aware | 47,708 (39.62%) | 53,481 (44.42%) | 19,219 (15.96%) |
| tail_only | 47,460 (39.27%) | 53,428 (44.21%) | 19,971 (16.52%) |

### 4.2 Head / mid / tail tokens

| Strategy | Head tokens | Mid tokens | Tail tokens |
|---|---:|---:|---:|
| full train | 81,249,594 (74.86%) | 22,329,365 (20.57%) | 4,961,651 (4.57%) |
| random | 22,461,338 (74.87%) | 6,177,301 (20.59%) | 1,361,279 (4.54%) |
| quality | 22,469,649 (74.90%) | 6,172,817 (20.58%) | 1,357,469 (4.52%) |
| diversity | 22,852,135 (76.17%) | 5,706,200 (19.02%) | 1,441,363 (4.80%) |
| tail_aware | 11,677,533 (38.93%) | 13,505,809 (45.02%) | 4,816,576 (16.06%) |
| tail_only | 11,590,586 (38.64%) | 13,448,345 (44.83%) | 4,960,923 (16.54%) |

**Up-weighting versus the natural train mix:**

| Strategy | Tail doc up-weight | Tail token up-weight |
|---|---:|---:|
| random | 0.995× | 0.993× |
| quality | 0.985× | 0.990× |
| diversity | 1.044× | 1.051× |
| tail_aware | **3.521×** | **3.512×** |
| tail_only | 3.645× | 3.618× |

Tail-aware selection raises the tail **token** share from 4.54% (random) to
16.06%, a 3.51× up-weight. This is the intended manipulation and it is large.
`tail_only` adds only 0.48 percentage points of token share over `tail_aware`
(16.54% vs 16.06%) while losing the quality filter.

### 4.3 Token share versus document share

The two must not be conflated. `quality` changes the document mix almost not at
all (tail doc share 4.47% vs random 4.51%) and the token mix almost not at all
(tail token share 4.52% vs 4.54%). Its effect is document *content quality*, not
topic rebalancing.

`tail_aware` changes both. Tail documents are 15.96% of selected documents and
16.06% of selected tokens: because tail documents are slightly longer on
average, their token share is marginally *above* their document share. The
practical statement is that tail-aware allocates about 16% of both documents and
tokens to tail categories, versus about 4.5% naturally.

### 4.4 Category and theme coverage

| Strategy | Primary categories covered | Effective categories (docs) | Themes covered (of 100) | Effective themes (tokens) | Smallest theme tokens | Fine clusters covered (of 1000) |
|---|---:|---:|---:|---:|---:|---:|
| full train | 152 | 13.99 | 100 | 91.35 | 65,627 | 1000 |
| random | 140 | 13.94 | 100 | 91.36 | 18,624 | 1000 |
| quality | 140 | 13.88 | 100 | 91.35 | 19,309 | 1000 |
| diversity | 145 | 13.39 | 100 | **93.38** | **8,805** | 1000 |
| tail_aware | 151 | **45.06** | 100 | 86.95 | 9,720 | 1000 |
| tail_only | 152 | **45.87** | 100 | 86.96 | 9,720 | 1000 |

All strategies cover every theme and every fine cluster at least once.
`tail_aware` and `tail_only` cover far more effective primary categories
(45 vs 14) because they deliberately spread mass across small arXiv fields.
`diversity` has the most even theme-token distribution (effective themes 93.4,
coefficient of variation 0.36) but no more theme coverage than the others.

---

## 5. Pairwise overlap

Upper triangle of document Jaccard, plus intersection size.

| A \ B | random | quality | diversity | tail_aware | tail_only |
|---|---:|---:|---:|---:|---:|
| **random** | 1.000 | 0.948 | 0.706 | 0.455 | 0.461 |
| **quality** | 0.948 | 1.000 | 0.726 | 0.464 | 0.450 |
| **diversity** | 0.706 | 0.726 | 1.000 | 0.432 | 0.420 |
| **tail_aware** | 0.455 | 0.464 | 0.432 | 1.000 | **0.946** |
| **tail_only** | 0.461 | 0.450 | 0.420 | **0.946** | 1.000 |

Concrete intersections:

| Pair | Intersection docs | Union docs | Jaccard |
|---|---:|---:|---:|
| random ∩ quality | 118,408 | 124,844 | 0.948 |
| random ∩ diversity | 101,126 | 143,166 | 0.706 |
| quality ∩ diversity | 102,655 | 141,357 | 0.726 |
| random ∩ tail_aware | 75,756 | 166,418 | 0.455 |
| quality ∩ tail_aware | 76,647 | 165,247 | 0.464 |
| diversity ∩ tail_aware | 73,240 | 169,694 | 0.432 |
| **tail_aware ∩ tail_only** | **117,299** | **123,968** | **0.946** |

**Interpretation.** `random` and `quality` are nearly the same dataset: they
share 94.8% of documents by Jaccard. The quality filter removes about 3,300
low-scoring documents and replaces them with about 3,100 better ones; 97.4% of
the *tokens* in `quality` are shared with `random`. `tail_aware` and `tail_only`
are also nearly the same dataset (0.946), differing mainly by the quality
filter. `diversity` is genuinely distinct from both (0.71–0.73).

This matters for the experimental review: a quality-vs-random comparison is a
comparison of a small document substitution, not of two independent corpora, so
its effect size is expected to be small.

---

## 6. Quality analysis

| Strategy | Mean quality | Median | p10 | p90 | Perfect (1.0) | Below 0.5 | Below 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full train | 0.96254 | — | — | — | 88.24% | 2.61% | — |
| random | 0.96216 | — | — | — | 88.16% | 2.66% | — |
| quality | 0.98198 | — | — | — | 90.56% | **0.00%** | — |
| diversity | 0.98108 | — | — | — | 90.15% | **0.00%** | — |
| tail_aware | 0.98084 | — | — | — | 89.97% | **0.00%** | — |
| tail_only | 0.95958 | — | — | — | 87.43% | 2.85% | — |

Quality by group (mean score / % below 0.5):

| Strategy | Head | Mid | Tail |
|---|---|---|---|
| random | 0.9630 / 2.58% | 0.9611 / 2.75% | 0.9529 / 3.53% |
| quality | 0.9822 / 0.00% | 0.9816 / 0.00% | 0.9794 / 0.00% |
| diversity | 0.9815 / 0.00% | 0.9801 / 0.00% | 0.9778 / 0.00% |
| tail_aware | 0.9824 / 0.00% | 0.9805 / 0.00% | 0.9779 / 0.00% |
| tail_only | 0.9636 / 2.52% | 0.9596 / 2.82% | 0.9501 / 3.70% |

**Does quality selection select higher-scoring documents? Yes.**
The quality-filtered strategies have mean score 0.981–0.982 and *zero* documents
below the 0.5 threshold, versus 2.6% for random. The effect is uniform across
head, mid, and tail.

The cleanest evidence is the symmetric difference between `quality` and
`random`:

| Set | Docs | Mean quality score |
|---|---:|---:|
| `quality` only | 3,078 | **0.9820** |
| `random` only | 3,358 | **0.2633** |

The documents that `quality` excludes are, on average, very low scoring
(0.263), so the filter is doing what it claims. `tail_only` intentionally does
not use the filter and retains 2.85% low-quality documents.

**Does quality selection introduce topic bias? No meaningful bias.**
The largest category doc-share difference versus `random` is 0.12 percentage
points (`cs.AI`: 9.90% vs 10.02%). The tail document share moves from 4.51% to
4.47%. Quality selection is essentially topic-neutral.

---

## 7. Diversity analysis

The project's operational diversity measure is the evenness of the token
distribution over the 100 text themes, plus fine-cluster coverage.

| Strategy | Effective themes (tokens) | Theme token CV | Smallest theme tokens | Top theme token share | Fine clusters |
|---|---:|---:|---:|---:|---:|
| random | 91.36 | 0.4336 | 18,624 | — | 1000 |
| quality | 91.35 | 0.4344 | 19,309 | — | 1000 |
| diversity | **93.38** | **0.3583** | 8,805 | — | 1000 |
| tail_aware | 86.95 | 0.5517 | 9,720 | — | 1000 |
| tail_only | 86.96 | 0.5512 | 9,720 | — | 1000 |

**Does diversity selection reduce redundancy? Partially.**
It raises effective theme count from 91.35 to 93.38 (out of 100) and reduces the
theme-token coefficient of variation from 0.434 to 0.358 — about a 17% reduction
in dispersion. It also raises the tail doc share slightly (4.73% vs 4.51%).

However, the absolute improvement is modest: 2.03 additional effective themes
out of 100, and **no** increase in theme coverage (all strategies cover 100
themes). The smallest theme still receives only 8,805 tokens, down from 19,309
in `quality`, because equalising themes necessarily starves the smallest ones
relative to a larger minimum. Diversity selection also reaches *fewer* effective
primary categories (13.4 vs 13.9) — it balances text themes, not arXiv fields.

**Duplicates.** After exact-text deduplication in Part 1, every subset has
**zero** exact-duplicate texts. Near-duplicate rate is not recomputed here
because the MinHash signatures were not persisted; the original
`reports/quality_report.json` documents the near-duplicate rule and applies it
only to the quality pool. This is a gap, not evidence of absence.

---

## 8. Tail representation

- Tail-aware and tail-only each cover **all 118** tail primary categories.
- Random and quality already cover all 118 tail categories, but with only
  ~4.5% of tokens; the tail is present but under-trained.
- Tail token share: random 4.54%, quality 4.52%, diversity 4.80%,
  tail_aware 16.06%, tail_only 16.54%.
- Tail-aware up-weights tail tokens **3.51×** relative to the natural mix.
- At the category level, tail-aware applies a roughly uniform 3.62× lift to
  small fields (e.g. `physics.plasm-ph` 0.032% → 0.116% of tokens,
  `gr-qc` 0.018% → 0.065%, `cond-mat.supr-con` 0.0038% → 0.0137%).

Because validation token mass is concentrated in head categories (head 75.6%,
mid 20.0%, tail 4.45%), this reallocation is large in training but touches only
a small fraction of validation tokens. That asymmetry is central to the results
review.

---

## 9. Findings and caveats

**Measured findings**

1. All five token files are correct and internally consistent (0 length
   mismatches, 0 double EOTs, exact 30M-token budgets).
2. `quality` genuinely selects higher-scoring documents and removes ~2.6%
   low-quality content with no meaningful topic bias.
3. `tail_aware` triples tail token exposure (4.54% → 16.06%) and covers all 118
   tail categories.
4. `tail_only` adds almost nothing over `tail_aware` on tail exposure (+0.48 pp
   tokens) and gives up the quality filter (2.85% low-quality documents).
5. `diversity` measurably evens theme-token allocation but by a modest amount
   (+2.0 effective themes out of 100) and does not increase theme coverage.
6. `random` and `quality` overlap at Jaccard 0.948; `tail_aware` and `tail_only`
   at 0.946. These pairs are nearly the same dataset.

**Caveats**

- The original `quality.parquet`, `token_counts.parquet`, and subset parquet
  files are missing; scores and lengths were recomputed from canonical inputs
  and cross-checked, not read from the original selection outputs.
- Near-duplicate statistics after selection could not be recomputed because
  MinHash signatures were not persisted.
- "Diversity" here means text-theme evenness as implemented in Part 2. It is not
  embedding-space coverage or a novelty metric, and it should not be described
  as such.
- Document counts and token counts are reported separately throughout; a
  strategy that selects more rare-topic documents does not automatically spend
  proportionally more tokens on them unless token share also rises.

**Bottom line.** The five strategies are not five independent corpora. Two pairs
are near-duplicates (`random`/`quality`, `tail_aware`/`tail_only`), and the
substantive manipulations are: quality filtering (small, targeted document
substitution) and tail rebalancing (large, 3.5× token reallocation). Diversity is
a real but modest theme-evenness change. Any claim that "five strategies were
compared" should be qualified accordingly.
