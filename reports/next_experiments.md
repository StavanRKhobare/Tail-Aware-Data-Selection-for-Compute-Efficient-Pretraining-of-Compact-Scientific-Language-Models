# Next Experiments

**Date:** 2026-11-10
**Status:** proposal only. Nothing in this document has been executed.
**Inputs:** `dataset_selection_audit.md`, `experimental_results_review.md`,
`compute_efficiency_review.md`.

---

## 1. What the audit says, and what it implies

| Finding | Strength | Implication |
|---|---|---|
| `quality` beats `random` on overall val loss in both seeds and all three topic groups | consistent, ~0.010–0.013 nats | Worth confirming with more seeds before claiming |
| `tail_aware` improves tail loss ~0.104 nats and tail PPL ~9.9%, but costs head ~0.023 nats | large, consistent, mechanistically explained | A real trade-off; needs a prespecified way to weigh it |
| `tail_only` adds no tail benefit over `tail_aware` and is worse overall | consistent | Drop it from the main claim; keep as ablation |
| `diversity` does not improve loss | consistent | Do not feature it in the main claim |
| `random`/`quality` overlap Jaccard 0.948; `tail_aware`/`tail_only` 0.946 | measured | Effects are small because the corpora are similar |
| Validation loss still falling at 60M tokens (best step = last step) | measured | Longer training is justified |
| Only 2 training seeds | design limitation | Statistical claims are not supported yet |
| Compute cost 12 min / 60M-token run | measured | Replication is cheap; design, not compute, is the constraint |

**Priority order:** seeds → longer training → larger model.

Rationale: additional seeds are the cheapest way to turn "consistent in two runs"
into a defensible effect estimate. Longer training is justified by the measured
curve and is still cheap. A larger model is the most expensive and answers a
different question (does the selection effect survive more capacity), so it
should come last and only after the effect is established at the current scale.

---

## 2. Experiment 1 — Additional seeds (recommended next)

### Research question
Is the quality-filter gain and the tail-aware head/tail trade-off stable across
training runs, or an artefact of seeds 42 and 43?

### Design

| Item | Value |
|---|---|
| Independent variable | training seed (5 levels) |
| Controlled | architecture 17.1M, tokenizer, 60M-token budget, all hyperparameters, subset definitions, validation split |
| Strategies | all five, unchanged |
| Seeds | 42, 43 (existing) + **44, 45, 46** (new) |
| New runs | 15 |
| Total runs | 25 |
| Primary metric | final validation loss at 60M tokens |
| Secondary (prespecified) | head/mid/tail validation loss; tail perplexity |
| Analysis | paired differences by seed; mean ± std; hierarchical bootstrap over seeds and validation documents |

### Compute
- 15 × 12.0 min = **3.0 GPU-hours** (measured throughput).
- Storage: checkpoints are small relative to the token files; no new data
  selection is needed.

### Conclusions this can support
- Whether `quality` > `random` holds in 5/5 seeds.
- A seed-level confidence interval for the quality gain and the tail-aware
  trade-off.
- Whether `tail_only`'s seed-43 collapse is typical or an outlier.

### Conclusions this cannot support
- Any claim about longer training or larger models.
- A population-level significance claim about "data selection strategies" in
  general; five seeds still describe this corpus and this architecture.

### Analysis change required
Report a **paired seed-level** interval, and state explicitly that it combines
two sources of variation (training seed and validation sampling). A simple
`mean ± std` over five seeds is acceptable but should be accompanied by the
per-seed paired differences.

---

## 3. Experiment 2 — Longer training (120M tokens)

### Research question
Do the strategy rankings and the tail-aware trade-off persist when models are
trained past the point where validation loss has flattened?

### Design

| Item | Value |
|---|---|
| Independent variable | token budget: 60M vs 120M |
| Controlled | architecture, tokenizer, seeds, hyperparameters, subsets, validation split |
| Strategies | all five |
| Seeds | 42, 43, 44 (3 seeds) |
| New runs | 15 at 120M |
| Primary metric | final validation loss at 120M |
| Secondary | loss at matched 60M checkpoint vs final; tail loss; tail PPL |

### Compute
- 24.0 min per 120M run (measured-throughput estimate).
- 15 runs = **6.0 GPU-hours**.

### Reduced-budget variant
If only the three informative strategies are run (`random`, `quality`,
`tail_aware`), 3 seeds × 3 strategies = 9 runs = **3.6 GPU-hours**. This is the
better first step; `diversity` and `tail_only` have already failed to show a
benefit and can wait.

### Evidence that justifies it
Last-segment improvement was 0.046–0.049 nats per ~10.8M tokens, and every run's
best step was its final step. The curve is concave but not flat.

### Conclusions this can support
- Whether the quality gain shrinks, persists, or grows with more tokens.
- Whether the head/tail trade-off narrows as both head and tail get more updates.
- Whether 60M was simply too early to rank strategies.

### Conclusions this cannot support
- Compute-optimal behaviour. 120M tokens for 17.1M parameters is still ~7
  tokens/parameter, far below Chinchilla's ~20.
- Anything about model scale.

### Important control
Evaluate every 120M run at the **same 60M-token checkpoint** as well as at the
end, so the budget effect is separated from the checkpoint-selection effect.

---

## 4. Experiment 3 — Larger model (~37M parameters)

### Research question
Does the tail-aware benefit, and its head-category cost, persist when the model
has more capacity to represent rare topics?

This is the scientific reason for scaling. Simply obtaining lower loss is not a
research contribution.

### Proposed configuration (exact parameter count)

| Layers | Heads | Width | Context | Parameters | × current |
|---:|---:|---:|---:|---:|---:|
| 9 | 8 | 512 | 512 | **37,023,232** | 2.16× |

Alternative: 8 layers / 8 heads / 512 width = 33,870,848 parameters (1.98×).
The 9-layer option sits inside the requested 35–40M band.

### Design (when eventually run)

| Item | Value |
|---|---|
| Independent variable | model size: 17.1M vs 37.0M |
| Controlled | tokenizer, subsets, seeds, token budget, effective batch size, schedule shape |
| Strategies | `random`, `quality`, `tail_aware` (drop the two dominated ones) |
| Seeds | 42, 43, 44 |
| Token budget | 120M (so the larger model is not even more undertrained than the small one) |
| New runs | 9 |
| Primary metric | final validation loss at 120M |
| Secondary | tail loss, tail PPL, head loss |

### Compute estimate (must be piloted)

- Linear-scaling estimate: 38,585 tokens/s → 120M / 38,585 = **51.8 min per run**.
- 9 runs ≈ **7.8 GPU-hours**, plus a 100-step pilot.
- Memory: weights and optimizer state are small, but activations at batch
  32 × 512 are likely to exceed the 4.7 GB currently used. Expect to reduce
  micro-batch to 8–16 with gradient accumulation; this preserves tokens/step but
  changes the step-time profile, so the estimate is unreliable until measured.

### Required pilot before committing
Run 100 steps at the target micro-batch, measure tokens/s and peak
`max_memory_allocated` and `max_memory_reserved`, then recompute the schedule.
Do not extrapolate linearly without this check.

### Conclusions this can support
- Whether the tail-aware trade-off is a small-model artefact.
- Whether more capacity converts tail exposure into larger tail gains.

### Conclusions this cannot support
- Chinchilla-optimal scaling claims; even 120M tokens is far below compute-optimal
  for 37M parameters.
- Any claim that a larger model is "better" in general; the comparison is
  conditional on the fixed budget.

---

## 5. Analyses to change before any new run

These are cheap and prevent post-hoc storytelling:

1. **Preregister the primary and secondary metrics.** Primary = final validation
   loss at the fixed budget. Secondary = tail loss / tail PPL. State in advance
   that tail-aware is expected to trade head for tail, so a single overall number
   is not the whole result.
2. **Report both overall and per-group loss for every strategy.** Never report
   tail loss alone or overall loss alone.
3. **Use paired seed-level comparisons.** Pair by seed, report per-seed
   differences and a seed-level interval; do not run an unpaired test across
   strategies.
4. **Keep the test split untouched until Part 4.** All of the above uses
   validation only.
5. **Add the missing instrumentation** listed in `compute_efficiency_review.md`
   (separate train/val timers, reserved memory, GPU utilization/power, per-step
   latency).
6. **Fix the checkpoint-selection rule.** `ckpt_best.pt` is selected on the same
   validation split used for analysis, which mildly biases levels. For future
   runs, either preselect the final checkpoint or hold out a small model-selection
   slice.

---

## 6. What not to do

- **Do not add more strategies.** Five already collapse into roughly three
  distinct corpora; adding a sixth would not resolve the inference problem.
- **Do not run `tail_only` and `diversity` at larger scale first.** Both failed
  to show a benefit at 60M; spend the budget on seeds and token horizon instead.
- **Do not scale the model before the effect is established.** A larger model
  with two seeds would repeat the current statistical weakness at 2× the cost.
- **Do not re-tune hyperparameters per strategy.** That would break the
  equal-compute, single-variable design.

---

## 7. Sequencing and total budget

| Step | Runs | Estimate | Cumulative |
|---|---:|---:|---:|
| 1. Seeds 44–46, all strategies, 60M | 15 | 3.0 h | 3.0 h |
| 2. 120M, 3 strategies, seeds 42–44 | 9 | 3.6 h | 6.6 h |
| 3. Pilot 37M model (100 steps) | 1 | ~0.5 h | 7.1 h |
| 4. 37M, 120M tokens, 3 strategies, 3 seeds | 9 | ~7.8 h (estimate) | ~14.9 h |

Step 1 alone would materially strengthen the paper. Steps 1–2 answer "is the
effect real and does it survive more training". Step 4 answers "does it survive
more capacity" and should be contingent on Steps 1–2 showing a stable effect.

---

## 8. The single recommended next experiment

**Replicate all five strategies at 60M tokens with three additional training
seeds (44, 45, 46).**

Why this one:

- It is the cheapest experiment that fixes the study's main weakness: all current
  claims rest on two training runs.
- It requires no new data selection, no architecture change, and no new
  infrastructure; the exact same command works with a different `--seed`.
- It directly tests whether the two robust-looking findings (quality gain,
  tail-aware trade-off) hold in 5/5 seeds.
- At ~3 GPU-hours it costs less than a single 120M-token campaign and can be run
  immediately.

Only after that should the budget be extended to 120M tokens, and only after
that should model scaling be attempted.
