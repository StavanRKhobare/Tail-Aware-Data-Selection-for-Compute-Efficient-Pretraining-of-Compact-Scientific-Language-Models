# Experimental Results Review

**Date:** 2026-11-10
**Scope:** the ten existing Part 3 runs (5 strategies × seeds 42, 43). No model was
retrained, and no test-split data was read.
**Script:** `audit_results.py` (read-only evaluation).
**Outputs:** `reports/experimental_results_*.json|csv`.

---

## 1. Metric definition and provenance

`PART3.md` defines the primary metric as **final validation loss at the fixed
60M-token budget**. That definition is respected here. Head/mid/tail and
per-category losses are reported as **diagnostic** outcomes; they were not
preregistered as primary and must not be presented as if they were.

Evaluation recomputes loss from `ckpt_best.pt` on `data/subsets/val.bin` using
non-overlapping 512-token windows: 11,697 windows, 5,988,864 scored targets over
24,401 validation documents. Recomputed overall loss matched each checkpoint's
stored `val_loss` to within 5×10⁻⁴ (see `experimental_results_audit.json`).
Perplexity is `exp(loss)` throughout, computed from the same loss.

Validation composition: head 75.6% of tokens, mid 20.0%, tail 4.45%.

---

## 2. Consolidated results

### 2.1 Strategy summary (mean ± std across two training seeds)

| Strategy | Final val loss | Final val ppl | Head loss | Mid loss | Tail loss | Tail ppl |
|---|---:|---:|---:|---:|---:|---:|
| **quality** | **3.64656 ± 0.00020** | **38.342** | **3.60716** | 3.72950 | 3.94315 | 51.581 |
| random | 3.65842 ± 0.00196 | 38.800 | 3.61899 | 3.74144 | 3.95531 | 52.212 |
| tail_aware | 3.66224 ± 0.00282 | 38.949 | 3.64174 | **3.69768** | **3.85141** | **47.059** |
| diversity | 3.66343 ± 0.00476 | 38.995 | 3.62212 | 3.75606 | 3.94903 | 51.886 |
| tail_only | 3.67153 ± 0.01199 | 39.313 | 3.65098 | 3.70776 | 3.85792 | 47.368 |

Ranking on the primary metric: `quality` < `random` < `tail_aware` <
`diversity` < `tail_only`.

### 2.2 Per-seed results

| Strategy | Seed | Overall | PPL | Head | Mid | Tail | Tail PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 42 | 3.65704 | 38.746 | 3.61759 | 3.74021 | 3.95346 | 52.116 |
| random | 43 | 3.65980 | 38.854 | 3.62039 | 3.74266 | 3.95716 | 52.308 |
| quality | 42 | 3.64670 | 38.348 | 3.60732 | 3.72941 | 3.94400 | 51.625 |
| quality | 43 | 3.64642 | 38.337 | 3.60701 | 3.72959 | 3.94230 | 51.537 |
| diversity | 42 | 3.66679 | 39.126 | 3.62538 | 3.75943 | 3.95409 | 52.148 |
| diversity | 43 | 3.66007 | 38.864 | 3.61885 | 3.75270 | 3.94398 | 51.624 |
| tail_aware | 42 | 3.66025 | 38.871 | 3.63973 | 3.69550 | 3.85047 | 47.015 |
| tail_aware | 43 | 3.66424 | 39.026 | 3.64375 | 3.69985 | 3.85235 | 47.103 |
| tail_only | 42 | 3.66305 | 38.980 | 3.64257 | 3.69863 | 3.85114 | 47.047 |
| tail_only | 43 | 3.68001 | 39.647 | 3.65938 | 3.71689 | 3.86471 | 47.689 |

Seed spread is small for `quality` (0.0003) and `random` (0.0028), but large for
`tail_only` (0.0170). `tail_only` seed 43 is the worst single run in the study.

### 2.3 Paired differences by training seed

Negative = first strategy has lower loss.

| Comparison | Seed | Overall | Head | Mid | Tail |
|---|---:|---:|---:|---:|---:|
| quality − random | 42 | **−0.01034** | −0.01027 | −0.01080 | −0.00946 |
| quality − random | 43 | **−0.01338** | −0.01338 | −0.01307 | −0.01486 |
| tail_aware − random | 42 | +0.00321 | +0.02214 | −0.04471 | **−0.10299** |
| tail_aware − random | 43 | +0.00444 | +0.02336 | −0.04281 | **−0.10481** |
| tail_only − tail_aware | 42 | +0.00280 | +0.00284 | +0.00313 | +0.00067 |
| tail_only − tail_aware | 43 | +0.01577 | +0.01563 | +0.01704 | +0.01236 |
| diversity − random | 42 | +0.00976 | +0.00779 | +0.01921 | +0.00062 |
| diversity − random | 43 | +0.00026 | −0.00153 | +0.01004 | −0.01317 |

The signs are consistent across both seeds for the four substantive
comparisons. Only the *magnitudes* differ.

### 2.4 Validation-document bootstrap

2,000 paired bootstraps over validation documents, per training seed, fixed
checkpoints. This quantifies **evaluation sampling uncertainty**, not training
seed uncertainty.

| Comparison | Seed | Scope | Observed | 95% CI |
|---|---:|---|---:|---|
| quality − random | 42 | overall | −0.01034 | [−0.01093, −0.00970] |
| quality − random | 43 | overall | −0.01338 | [−0.01399, −0.01279] |
| tail_aware − random | 42 | overall | +0.00321 | [+0.00208, +0.00436] |
| tail_aware − random | 43 | overall | +0.00444 | [+0.00332, +0.00549] |
| tail_aware − random | 42 | tail | −0.10299 | [−0.10875, −0.09730] |
| tail_aware − random | 43 | tail | −0.10481 | [−0.11083, −0.09912] |
| tail_aware − random | 42 | head | +0.02214 | [+0.02106, +0.02335] |
| tail_aware − random | 43 | head | +0.02336 | [+0.02216, +0.02443] |
| tail_only − tail_aware | 42 | overall | +0.00280 | [+0.00222, +0.00341] |
| tail_only − tail_aware | 43 | overall | +0.01577 | [+0.01515, +0.01641] |
| diversity − random | 42 | overall | +0.00976 | [+0.00908, +0.01046] |
| diversity − random | 43 | overall | +0.00026 | [−0.00046, +0.00098] |

Every CI above excludes zero except `diversity − random` at seed 43. But **these
intervals are conditional on the two trained checkpoints**. They do not license
the claim "the strategies differ significantly", because the dominant
uncertainty is which training run you got, and that is sampled only twice.

---

## 3. Head / mid / tail findings

**Tail-aware works exactly where it is designed to work.**

- Tail loss falls from 3.9554 (random mean) to 3.8514, a −0.104 nat improvement
  per seed (CI ≈ [−0.109, −0.097] and [−0.111, −0.099]).
- Tail perplexity falls from 52.21 to 47.06, about 9.9% lower.
- Mid loss also improves: 3.7414 → 3.6977 (−0.044).
- Head loss worsens: 3.6190 → 3.6417 (+0.023).
- Because head is 75.6% of validation tokens, the head penalty slightly
  outweighs the mid/tail gains in the overall average: +0.0032 and +0.0044.

At the category level, of 92 tail categories with validation tokens, **90 improve**
under tail-aware (median −0.101 nats, token-weighted −0.104). The two exceptions
are `cs.MM` (+0.011) and `physics.ed-ph` (+0.007), both tiny. Largest tail gains:

| Category | Val tokens | Random | Tail-aware | Δ |
|---|---:|---:|---:|---:|
| cond-mat.str-el | 805 | 4.014 | 3.694 | −0.320 |
| hep-lat | 631 | 3.523 | 3.279 | −0.244 |
| gr-qc | 1,047 | 3.856 | 3.618 | −0.238 |
| nucl-th | 472 | 4.321 | 4.100 | −0.221 |
| astro-ph.CO | 3,682 | 3.973 | 3.759 | −0.214 |

Among the six head categories, four worsen and two improve. Largest head
regressions: `cs.CL` +0.060, `cs.CV` +0.031, `cs.LG` +0.013, `cs.AI` +0.010.

**This is a real trade-off, not a win.** Tail-aware buys tail robustness by
spending head-category accuracy. Any summary that reports only tail loss, or
only overall loss, is incomplete.

---

## 4. Validation-loss progression

Validation is evaluated every 250 steps (4.096M tokens). "At N M" means the last
evaluation at or below N million tokens.

| Strategy | Seed | @30M | @40M | @50M | Final (60M) | Δ 50M→60M |
|---|---:|---:|---:|---:|---:|---:|
| random | 42 | 3.9606 | 3.8240 | 3.7056 | 3.6570 | 0.0486 |
| random | 43 | 3.9546 | 3.8241 | 3.7066 | 3.6598 | 0.0468 |
| quality | 42 | 3.9417 | 3.8116 | 3.6928 | 3.6467 | 0.0461 |
| quality | 43 | 3.9428 | 3.8092 | 3.6942 | 3.6464 | 0.0477 |
| tail_aware | 42 | 3.9642 | 3.8270 | 3.7089 | 3.6602 | 0.0487 |
| tail_aware | 43 | 3.9739 | 3.8340 | 3.7119 | 3.6642 | 0.0476 |
| diversity | 42 | 3.9699 | 3.8351 | 3.7152 | 3.6668 | 0.0484 |
| diversity | 43 | 3.9565 | 3.8258 | 3.7056 | 3.6601 | 0.0456 |
| tail_only | 42 | 3.9661 | 3.8289 | 3.7100 | 3.6630 | 0.0470 |
| tail_only | 43 | 3.9822 | 3.8492 | 3.7277 | 3.6800 | 0.0477 |

All ten runs were still improving at the final step; `best_step` equals 3662 for
every run. The last ~10.8M tokens still bought 0.046–0.049 nats. The curves are
concave but far from flat, which is the empirical case for a longer budget.

The ranking is already stable at 30M (`quality` best, `tail_aware`/`tail_only`
behind `random` overall, `diversity` mid-pack), so the 60M result is not an
artefact of a single late evaluation point.

---

## 5. Direct answers

### 5.1 Does quality selection consistently beat random?

**Yes, on this evidence, in both seeds and every topic group.** Overall
−0.0103 and −0.0134 nats; head −0.010/−0.013; mid −0.011/−0.013; tail
−0.009/−0.015. The seed spread of `quality` (0.0003) is an order of magnitude
smaller than the effect. The paired validation bootstrap excludes zero for all
groups in both seeds. This is the most robust finding in the study.

Caveat: the two subsets overlap at Jaccard 0.948, so the effect is "replacing
~2.7% low-quality documents raises validation likelihood", not "a different
corpus". That is still a legitimate, and compute-efficient, result.

### 5.2 Does tail-aware consistently improve tail performance?

**Yes, strongly and consistently.** −0.103 and −0.105 nats on tail; tail PPL
−9.9%; 90 of 92 tail categories improve. It also improves mid by ~0.044 nats.
It consistently costs head accuracy (+0.022/+0.023).

### 5.3 Does tail-only add anything over tail-aware?

**No measurable benefit, and it is worse overall.** Tail difference is +0.0007
(CI crosses zero) at seed 42 and +0.0124 (worse) at seed 43. Overall it is worse
by +0.0028 and +0.0158, and it is the most seed-sensitive strategy
(std 0.0120). Dropping the quality filter does not buy additional tail
performance; it mainly adds variance. `tail_only` is best interpreted as a
negative ablation: the quality filter is worth keeping.

### 5.4 Does diversity improve anything measurable?

**Only theme evenness, and not downstream loss.** It raises effective themes from
91.35 to 93.38 and cuts theme-token CV from 0.434 to 0.358, but validation loss
is equal to random at seed 43 (+0.0003, CI includes zero) and worse at seed 42
(+0.0098), and worse than quality at both seeds (+0.020/+0.014). No group
improves consistently. There is no evidence that the diversity objective helps
this model at this scale and budget.

### 5.5 Are the differences robust enough for a research claim?

Distinguish three levels:

| Claim | Supported? |
|---|---|
| quality > random on this corpus, architecture, and budget | **Yes**, consistent across 2 seeds and all groups |
| tail_aware trades head accuracy for tail accuracy | **Yes**, large and consistent |
| tail_only ≤ tail_aware | **Yes**, consistent and quality-filter argument is clear |
| diversity helps | **No** |
| any strategy is significantly better in a population-of-training-runs sense | **Not established** with n = 2 |

With two seeds, a paired t-test has essentially no power, and seed standard
deviations are themselves estimated from two points. The correct statement is:
"effects are consistent across the two seeds we ran", not "statistically
significant". No p-values should be reported for strategy comparisons.

---

## 6. Limitations

1. **Two training seeds.** Seed uncertainty is not resolved; the bootstrap does
   not address it.
2. **High subset overlap.** `random`/`quality` (Jaccard 0.948) and
   `tail_aware`/`tail_only` (0.946) are near-duplicates, which mechanically
   shrinks effect sizes and makes "five independent strategies" an overstatement.
3. **Undertrained regime.** 60M tokens for 17.1M parameters is ~3.5
   tokens/parameter, far below compute-optimal. Rankings may change with longer
   training; the curves are still descending.
4. **Primary metric is overall loss.** Tail-aware is *designed* to lose on this
   metric. Reporting only the primary metric would hide the intended effect;
   reporting only tail loss would hide the cost.
5. **Per-category losses are diagnostic.** Categories have very different token
   counts (hundreds to ~1.8M), so category means are not equally precise and
   were not preregistered.
6. **Validation reuse.** The same validation split was used for checkpoint
   selection (`ckpt_best.pt`) and for this analysis, so these numbers are mildly
   optimistic in level, though the comparison remains paired.
7. **Test split untouched.** No test-set number exists yet; Part 4 owns it.

---

## 7. Verdict

- `quality` is the best strategy on the preregistered primary metric and the
  only strategy with a consistent, same-direction gain in every topic group.
- `tail_aware` successfully and substantially improves tail and mid performance
  (tail PPL 52.2 → 47.1) at a small, consistent overall cost driven by head
  categories.
- `tail_only` is dominated by `tail_aware`: no tail benefit, worse overall, more
  seed variance, and lower data quality.
- `diversity` does not improve any loss metric; its only measured effect is
  modest theme-token evenness.
- **Do not claim statistical significance from two seeds.** Do claim a
  consistent head/tail trade-off and a consistent quality-filter gain, both of
  which are large relative to the observed seed spread.
