# Five-Seed Results Review

**Date:** 2026-11-10
**Scope:** 25 runs — 5 strategies × seeds 42, 43, 44, 45, 46, all at 60M tokens.
**Status:** supersedes the strategy-level conclusions in
`experimental_results_review.md` (which used only seeds 42 and 43). That file is
left in place as the historical two-seed analysis.
**Scripts:** `audit_results.py --seeds 42 43 44 45 46 --tag 5seed`,
`analyze_seeds.py`, `audit_compute.py --seeds 42 43 44 45 46 --tag 5seed`.
**New artifacts:** `reports/5seed_*`.

---

## 0. Headline correction

> The two-seed result that `quality` beats `random` **did not replicate**.
> Across five seeds, `quality` is slightly *worse* than `random` on the
> preregistered primary metric (mean 3.6542 vs 3.6514), it wins in only 2 of 5
> seeds, and the paired seed-level interval spans zero (p ≈ 0.72).

The tail-aware finding **did replicate**: tail loss improves in 5/5 seeds by
0.069–0.105 nats, and head loss worsens in 5/5 seeds. The overall cost of
tail-aware is small but positive in 5/5 seeds.

This is exactly the outcome the earlier review warned about: two seeds cannot
support a strategy claim, and the two seeds available were unusually favourable
to the quality filter.

---

## 1. Campaign verification

All 25 runs are valid and comparable:

| Check | Result |
|---|---|
| Runs present | 5 strategies × 5 seeds = 25 |
| Steps per run | 3,662 (all) |
| Tokens seen | 59,998,208 (all) |
| Model config identical | yes (25/25) |
| Optimizer/schedule identical | yes (25/25) |
| `max_steps`, `tokens_per_step`, `planned_tokens` identical | yes |
| Run identity (subset, seed, `train.bin` path) | matches in 25/25 |
| Validation evaluations per run | 16 |
| Validation loss finite | 25/25 |
| Validation loss monotone decreasing | 25/25 |
| Peak allocated memory | 4,703 MB in every run |
| Mean wall time | 12.01 min (range 11.98–12.08) |
| Mean throughput | 83,265 tokens/s |
| Gradient norms | 1.62–1.80, no spikes |

There is no evidence of a broken or partially completed run. The small
throughput difference between the earlier and later runs (83.4k → 83.0k
tokens/s on the `random`/`quality` subsets) is consistent with mild thermal
variation on a laptop GPU and does not affect loss.

---

## 2. Five-seed results

### 2.1 Strategy summary (mean ± SD across five training seeds)

| Strategy | Overall loss | Overall PPL | Head | Mid | Tail | Tail PPL |
|---|---:|---:|---:|---:|---:|---:|
| **random** | **3.65137 ± 0.00796** | **38.528** | **3.61193** | 3.73454 | 3.94773 | 51.819 |
| quality | 3.65418 ± 0.01044 | 38.638 | 3.61471 | 3.73741 | 3.95078 | 51.979 |
| diversity | 3.66347 ± 0.00445 | 38.997 | 3.62210 | 3.75625 | 3.94942 | 51.906 |
| tail_aware | 3.66772 ± 0.00802 | 39.164 | 3.64728 | **3.70288** | **3.85714** | **47.331** |
| tail_only | 3.66888 ± 0.00980 | 39.210 | 3.64841 | 3.70456 | 3.85637 | 47.295 |

Ranking on the primary metric: `random` < `quality` < `diversity` <
`tail_aware` < `tail_only`.

Note the seed SD is now **0.004–0.010**, i.e. 2–5× larger than the SD estimated
from the first two seeds (0.002–0.005). This alone explains why the two-seed
ranking was unstable.

### 2.2 Per-seed overall loss

| Seed | random | quality | diversity | tail_aware | tail_only |
|---:|---:|---:|---:|---:|---:|
| 42 | 3.65704 | **3.64670** | 3.66679 | 3.66025 | 3.66305 |
| 43 | 3.65980 | **3.64642** | 3.66007 | 3.66424 | 3.68001 |
| 44 | **3.64093** | 3.65202 | 3.65736 | 3.66843 | 3.66644 |
| 45 | **3.65370** | 3.65384 | 3.66658 | 3.66459 | 3.65705 |
| 46 | **3.64539** | 3.67194 | 3.66656 | 3.68110 | 3.67786 |

Per-seed ranking (best → worst):

| Seed | Ranking |
|---:|---|
| 42 | quality > random > tail_aware > tail_only > diversity |
| 43 | quality > random > diversity > tail_aware > tail_only |
| 44 | random > quality > diversity > tail_only > tail_aware |
| 45 | random > quality > tail_only > tail_aware > diversity |
| 46 | random > diversity > quality > tail_only > tail_aware |

`random` is best in 3/5 seeds, `quality` in 2/5. The ordering of the other three
also changes. Only the *bottom* of the table is stable: `tail_aware` and
`tail_only` are 4th or 5th in every seed.

---

## 3. Seed-level paired inference

Differences are computed within each seed (same initialization and data-order
seed across strategies), then summarised over seeds. Negative = first strategy
has lower loss.

| Comparison | Metric | Mean Δ | SD | 95% t interval | p (paired t, df=4) | Seeds negative |
|---|---|---:|---:|---|---:|---:|
| quality − random | overall | +0.00281 | 0.01639 | [−0.01754, +0.02316] | 0.721 | 2/5 |
| quality − random | tail | +0.00305 | 0.01891 | [−0.02043, +0.02652] | 0.737 | 3/5 |
| tail_aware − random | overall | +0.01635 | 0.01452 | [−0.00168, +0.03438] | 0.066 | 0/5 |
| tail_aware − random | head | +0.03535 | 0.01441 | [+0.01745, +0.05324] | 0.005 | 0/5 |
| tail_aware − random | mid | −0.03166 | 0.01474 | [−0.04997, −0.01336] | 0.009 | 5/5 |
| tail_aware − random | tail | −0.09059 | 0.01563 | [−0.11000, −0.07119] | 0.0002 | 5/5 |
| tail_only − tail_aware | overall | +0.00116 | 0.00896 | [−0.00997, +0.01228] | 0.787 | 3/5 |
| tail_only − tail_aware | tail | −0.00077 | 0.00817 | [−0.01091, +0.00938] | 0.844 | 3/5 |
| diversity − random | overall | +0.01210 | 0.00786 | [+0.00234, +0.02186] | 0.026 | 0/5 |
| diversity − random | mid | +0.02171 | 0.00826 | [+0.01145, +0.03197] | 0.004 | 0/5 |
| diversity − quality | overall | +0.00929 | 0.00972 | [−0.00279, +0.02136] | 0.100 | 1/5 |

The paired t interval treats seeds as exchangeable replicates. With df = 4 it is
coarse; a percentile bootstrap over whole seeds gives similar intervals (see
`5seed_paired_seed_analysis.csv`). Neither should be read as a population-level
significance claim.

---

## 4. The correction in detail

The two-seed report found `quality − random` = −0.0103 and −0.0134, both with
validation-bootstrap CIs excluding zero, and concluded the effect was robust.
Adding three seeds gives:

```
seed 42: -0.01034
seed 43: -0.01338
seed 44: +0.01109
seed 45: +0.00014
seed 46: +0.02655
```

Mean +0.0028, SD 0.0164, p = 0.72, direction flips in 3/5 seeds.

What happened: seeds 42 and 43 happened to be the two seeds on which the
quality subset trained best relative to `random`. Seed 44 `random` was the
single best run in the whole campaign (3.6409), and seed 46 `quality` was the
worst quality run (3.6719). The original validation-document bootstrap was
correct but answered the wrong question — it held the checkpoints fixed and
asked about validation sampling, while the dominant uncertainty was which
training run you got.

**Therefore the correct statement is:** quality filtering removes genuinely
lower-quality training documents (mean score 0.263 for the excluded set) and is
topic-neutral, but at this scale and budget it produces **no detectable
improvement in validation loss**.

---

## 5. What did replicate: the tail-aware trade-off

| Seed | Δ tail loss | Δ mid loss | Δ head loss | Δ overall |
|---:|---:|---:|---:|---:|
| 42 | −0.10299 | −0.04471 | +0.02214 | +0.00321 |
| 43 | −0.10481 | −0.04281 | +0.02336 | +0.00444 |
| 44 | −0.07950 | −0.02126 | +0.04670 | +0.02750 |
| 45 | −0.09656 | −0.03840 | +0.03025 | +0.01089 |
| 46 | −0.06910 | −0.01114 | +0.05428 | +0.03571 |

- Tail improvement: **5/5 seeds**, mean −0.091 nats, p ≈ 0.0002.
- Mid improvement: **5/5 seeds**, mean −0.032 nats.
- Head cost: **5/5 seeds**, mean +0.035 nats.
- Overall cost: **5/5 seeds**, mean +0.016 nats, marginal p = 0.066.

The mechanism is exactly as designed: tail-aware reallocates ~3.5× more training
tokens to tail categories, improves the categories it up-weights, and pays for
it on the head categories that supply 75.6% of validation tokens. **The tail
gain is robust; the overall cost is robust in sign and marginal in magnitude.**

At category level (5-seed means):

- 90 of 92 tail categories improve (token-weighted −0.091 nats).
- 26 of 28 mid categories improve (−0.032 nats).
- 4 of 6 head categories worsen (+0.035 nats).
- Largest tail gains: `cond-mat.str-el` −0.273, `hep-lat` −0.228, `gr-qc` −0.222,
  `nucl-th` −0.221, `astro-ph.CO` −0.213.
- Largest head costs: `cs.CL` +0.072, `cs.CV` +0.044, `cs.LG` +0.026, `cs.AI` +0.021.

---

## 6. `tail_only` and `diversity`

**`tail_only` versus `tail_aware`:** indistinguishable. Overall +0.0012
(p = 0.79), tail −0.0008 (p = 0.84), 3/5 seeds in each direction. Dropping the
quality filter neither helps tail performance nor overall loss. The earlier
"tail_only is worst" impression came from seed 43 (+0.0158), which is an
outlier: seeds 44 and 45 favour tail_only slightly.

**`diversity`:** consistently worse than `random` — +0.0121 nats overall
(95% CI [+0.0023, +0.0219], 0/5 seeds better), driven mainly by mid categories
(+0.0217). Its only measured benefit remains theme-token evenness in the
training data, which does not translate into validation loss.

So the 5-seed ordering separates into two groups:

1. `random` and `quality` — statistically indistinguishable overall.
2. `diversity`, `tail_aware`, `tail_only` — worse overall by 0.012–0.017 nats.

Within group 2, tail_aware/tail_only buy a large tail improvement that
diversity does not.

---

## 7. What claims survive

| Claim | Status after 5 seeds |
|---|---|
| quality > random on overall val loss | **Refuted / not supported** |
| quality filter selects better documents | **Supported in the data**, but no loss benefit |
| tail_aware improves tail loss | **Supported, 5/5 seeds, ~0.09 nats** |
| tail_aware costs head accuracy | **Supported, 5/5 seeds, ~0.035 nats** |
| tail_aware costs overall loss | **Supported in sign (5/5), marginal magnitude** |
| tail_only beats tail_aware | **Not supported** |
| diversity improves loss | **Not supported; it is consistently worse** |
| any strategy is universally best | **Not supported** |
| effects are statistically significant at population level | **Not established** (5 seeds, one corpus, one architecture, one budget) |

The honest summary is: **data-selection strategy has a small effect on overall
validation loss at this scale, and the seed-to-seed variation is comparable to or
larger than the strategy differences — except for the tail-aware tail/head
trade-off, which is large and reproducible.**

---

## 8. Compute

The 25-run campaign is compute-matched and verified:

- 25 × 12.01 min ≈ **5.0 GPU-hours** total.
- 3,662 steps and 59,998,208 tokens per run.
- 4,703 MB peak allocated (57.9% of the 8 GB GPU).
- Validation overhead ≈ 16.4 s per run.

Full five-seed table: `reports/5seed_compute_efficiency_runs.csv`.

---

## 9. Implications for the next experiment

The five-seed result changes the priority. Replicating seeds is done; the open
question is no longer "is quality better?" but **"why does the quality gain
disappear, and does the tail trade-off persist with more training?"**

1. **Do not extend seeds further at 60M.** Five seeds already show the effect
   is below seed noise for the overall metric. More seeds would narrow an
   interval around approximately zero.
2. **Extend training to 120M tokens** for `random`, `quality`, and `tail_aware`
   at the same five seeds (15 runs, ~6.0 GPU-hours). Rationale: all 25 runs were
   still improving at the final step (0.046–0.050 nats in the last ~10.8M
   tokens), so a longer horizon could either reveal a quality-filter effect that
   needs more updates or confirm its absence.
3. **Keep the tail metric secondary but prespecified.** The tail gain is the one
   result strong enough to build on. Report it alongside overall loss, never
   instead of it.
4. **Consider a stronger quality manipulation if the 120M result is still null.**
   Removing 2.7% of documents is a small intervention; if the research question
   requires a quality effect, the filter itself may need to be more aggressive.
   That would be a new independent variable and must not be conflated with the
   current one.
5. **Do not scale the model yet.** A larger model would repeat the same
   statistical design at higher cost without first knowing whether the effects
   survive more tokens.

Recommended single next run:

```powershell
uv run --group train python src/train_all.py --seeds 42 43 44 45 46 --max-tokens 120000000 --subsets random quality tail_aware
```

That is 15 runs at ~24 min each, ≈ 6 GPU-hours. Use a distinct runs directory
(e.g. `--runs-dir runs_120m`) so the 60M checkpoints are preserved.

---

## 10. Verdict

- The five-seed campaign is complete, valid, and compute-matched.
- **The quality-selection advantage was a two-seed artefact and does not
  survive replication.**
- **Tail-aware selection reliably trades head accuracy for a large tail gain**;
  its small overall cost is consistent but marginal.
- `tail_only` is equivalent to `tail_aware`, and `diversity` is worse than
  `random`.
- No current strategy is a defensible "best overall" claim. The defensible claim
  is the tail/head trade-off, reported with its cost.
