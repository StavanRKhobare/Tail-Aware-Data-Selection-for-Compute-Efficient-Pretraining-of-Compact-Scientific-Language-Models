# Compute Efficiency Review

**Date:** 2026-11-10
**Scope:** measured runtime, throughput, and memory for the ten existing Part 3
runs, plus explicitly labelled estimates for future budgets.
**Script:** `audit_compute.py` (read-only).
**Outputs:** `reports/compute_efficiency_audit.json`,
`reports/compute_efficiency_runs.csv`,
`reports/compute_efficiency_strategy_summary.csv`,
`reports/compute_efficiency_model_estimates.csv`.

All numbers below are **measured** unless a line is explicitly labelled
"estimate".

---

## 1. Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| GPU memory | 8,123.4 MB (≈8 GB) |
| Compute capability | 12.0 |
| PyTorch | 2.14.1+cu130 |
| CUDA | 13.0 |
| Precision | bfloat16 autocast |
| Device count | 1 |

This is a single laptop-class GPU with 8 GB of memory. Absolute throughput will
not transfer to server GPUs; relative comparisons and memory headroom still
inform planning.

---

## 2. Equal-compute design verification

The ten `config.json` files were compared field by field. **No mismatches** were
found in model configuration, training hyperparameters, `max_steps`,
`tokens_per_step`, or `planned_tokens`.

| Quantity | Value (all ten runs) |
|---|---|
| Parameters | 17,135,616 |
| Layers / heads / width | 6 / 6 / 384 |
| Context length | 512 |
| Batch × grad-accum | 32 × 1 |
| Tokens per step | 16,384 |
| Steps | 3,662 |
| Planned tokens | 59,998,208 |
| Tokens seen | 59,998,208 |
| Token budget | 60M |
| Eval every | 250 steps (16 evals) |
| Log every | 10 steps (368 rows) |
| Save every | 500 steps |

Every run completed the full schedule. The compute-budget assumption in the
experimental design is verified, not assumed.

---

## 3. Measured per-run cost

| Strategy | Seed | Wall (s) | Wall (min) | Train segment (s) | Eval overhead (s) | Tokens/s (summary) | Peak alloc (MB) | s per 1M tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 42 | 719.37 | 11.99 | 702.98 | 16.39 | 83,404 | 4,703.0 | 11.99 |
| random | 43 | 719.81 | 12.00 | 703.39 | 16.42 | 83,352 | 4,703.0 | 12.00 |
| quality | 42 | 719.95 | 12.00 | 703.56 | 16.40 | 83,337 | 4,703.0 | 12.00 |
| quality | 43 | 719.93 | 12.00 | 703.56 | 16.38 | 83,339 | 4,703.0 | 12.00 |
| diversity | 42 | 719.58 | 11.99 | 703.20 | 16.39 | 83,379 | 4,703.0 | 11.99 |
| diversity | 43 | 719.53 | 11.99 | 703.15 | 16.38 | 83,385 | 4,703.0 | 11.99 |
| tail_aware | 42 | 719.60 | 11.99 | 703.21 | 16.39 | 83,377 | 4,703.0 | 11.99 |
| tail_aware | 43 | 719.81 | 12.00 | 703.42 | 16.39 | 83,353 | 4,703.0 | 12.00 |
| tail_only | 42 | 719.70 | 11.99 | 703.31 | 16.39 | 83,366 | 4,703.0 | 12.00 |
| tail_only | 43 | 719.67 | 11.99 | 703.27 | 16.40 | 83,369 | 4,703.0 | 11.99 |

**Aggregate measured values**

- Mean wall time: **719.70 s = 12.00 min** per 60M-token run.
- Mean training-segment time: ≈703.4 s; validation overhead ≈**16.4 s**
  (≈2.3% of wall time).
- Mean throughput: **83,366 tokens/s** (range 83,337–83,404).
- Peak allocated GPU memory: **4,703.0 MB**, i.e. **57.9% of 8,123 MB**.
- Total measured compute for the ten runs: ≈7,197 s ≈**2.0 GPU-hours**.

Throughput varies by only 0.08% across strategies, which is expected because
only the data file changes and model/batch shape is identical. Runtime is
therefore not a confound in the strategy comparison.

---

## 4. Is the comparison compute-matched?

Yes. All ten runs used the same number of steps, the same tokens per step, the
same batch shape, and the same evaluation cadence. Wall-time differences are
noise (<0.4 s). Any loss difference is attributable to the training data, not to
differing compute.

One caveat: strategies whose documents are on average slightly longer
(`tail_aware` 249.2 vs `random` 246.4 tokens/doc) still consume exactly the same
token budget; the number of *documents* seen differs by ~1.1%. This is the
correct equal-token design, but it means "same compute" is not "same number of
distinct documents".

---

## 5. Memory headroom

| Component | Approximate size | Basis |
|---|---:|---|
| bf16 weights (17.1M params × 2 B) | 34 MB | exact parameter count |
| fp32 master weights + AdamW moments + grads (≈16 B/param) | 274 MB | estimate |
| Peak **allocated** GPU memory (measured) | 4,703 MB | `torch.cuda.max_memory_allocated` |
| Remaining of 8 GB | ≈3,420 MB | arithmetic |

Activations at batch 32 × 512 tokens dominate the measured footprint. There is
enough headroom to increase batch size or slightly widen the model on this GPU,
but not enough for a 3–4× larger model at the same batch size without gradient
accumulation or smaller micro-batches.

`max_memory_reserved` was not logged, so fragmentation headroom is unknown; the
3.4 GB figure is an upper bound on usable additional allocation, not a guarantee.

---

## 6. Labelled estimates for future runs

### 6.1 120M-token budget (same architecture)

**Label: arithmetic estimate from measured mean throughput; no 120M run exists.**

| Quantity | Estimate |
|---|---:|
| Tokens per run | 120,000,000 |
| Wall per run | 1,439.4 s ≈ **24.0 min** |
| Steps per run | 7,324 (at 16,384 tokens/step) |
| 10 runs (5 strategies × 2 seeds) | **4.0 h** |
| 15 runs (5 strategies × 3 seeds) | **6.0 h** |
| 25 runs (5 strategies × 5 seeds) | **10.0 h** |

Assumes throughput stays at 83,366 tokens/s. Longer runs may show slight
thermal throttling on a laptop GPU, so treat these as lower bounds on wall time.
Validation overhead scales with the number of evaluations, not with tokens, so it
remains a small constant.

### 6.2 Larger model

**Label: parameter counts are exact (computed by instantiating `GPTConfig`);
throughput and memory extrapolations are linear-scaling estimates that must be
verified by a short pilot.**

| Layers | Heads | Width | Params | × current | Est. tokens/s | Est. 60M wall | Est. Adam state |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 384 | 17,135,616 | 1.00× | 83,366 (measured) | 12.0 min (measured) | 274 MB |
| 8 | 8 | 512 | 33,870,848 | 1.98× | 42,176 | 23.7 min | 542 MB |
| **9** | **8** | **512** | **37,023,232** | **2.16×** | **38,585** | **25.9 min** | **592 MB** |
| 10 | 8 | 512 | 40,175,616 | 2.34× | 35,557 | 28.1 min | 643 MB |

The 9-layer / 8-head / 512-width configuration gives **37.02M parameters**,
inside the requested 35–40M range. Linear parameter scaling is a crude proxy:
attention and MLP FLOPs do not scale exactly with parameter count, and memory
scaling is dominated by activations and batch size rather than weights. A
100-step pilot is required before committing to a full run.

The RTX 5050's 8 GB is likely to be the binding constraint for the 37M model at
batch 32 × 512. Expect to reduce micro-batch to 8–16 and use gradient
accumulation to preserve the effective batch size; that keeps token throughput
roughly constant but reduces per-step parallelism.

---

## 7. Missing instrumentation

The following were **not logged** and should be added before the next campaign:

1. **Separate training and validation timers.** The current script stores a
   single wall time plus `elapsed_s` from the last training log. Eval overhead
   was derived as `wall − elapsed`, which conflates the final checkpoint save
   and teardown with validation.
2. **GPU utilization and power/energy.** No `nvidia-smi` sampling, so
   utilization, clock throttling, and joules/token are unknown.
3. **Reserved vs allocated memory.** Only `max_memory_allocated` exists; add
   `max_memory_reserved` to measure fragmentation headroom.
4. **Per-step latency distribution.** Metrics are logged every 10 steps, so
   variance, stalls, and first-step warm-up are invisible.
5. **Host-side cost.** No CPU time, host RAM, dataloader time, or tokenization
   time. Part 2 tokenization and selection cost is entirely outside these runs.
6. **Checkpoint I/O time.** `save_every=500` writes 7 checkpoints per run; the
   I/O cost is not separated.
7. **Exact device and driver identity.** The GPU name is available, but driver
   version and power mode are not recorded, which matters for laptop hardware
   reproducibility.

---

## 8. Interpretation

- The experiment is **compute-matched by construction and verified by config
  comparison**. This is the strongest part of the Part 3 design.
- At 12 minutes per 60M-token run, the current budget is cheap on this hardware.
  A five-seed replication costs about 5 GPU-hours; a 120M-token, five-seed
  campaign costs about 10 GPU-hours. Compute is not the bottleneck for
  strengthening the study — the bottleneck is experimental design.
- Peak memory (4.7 of 8 GB) leaves limited room. Any larger model should be
  piloted for memory before a full campaign, and micro-batching will likely be
  needed.
- Runtime differences between strategies are negligible, so **time-to-target-loss**
  is a legitimate secondary metric if a target loss is defined in advance. It was
  not defined here, so it should not be introduced post hoc as a primary claim.

---

## 9. Verdict

- Equal-compute assumption: **verified**.
- Measured cost per run: **12.0 min, 59,998,208 tokens, 83,366 tokens/s,
  4,703 MB peak allocated**.
- Longer-training and larger-model figures: **estimates only**, clearly labelled,
  and not to be reported as measured results.
- Recommended instrumentation: separate train/val timers, reserved memory, GPU
  utilization/power, per-step latency, host-side cost.
