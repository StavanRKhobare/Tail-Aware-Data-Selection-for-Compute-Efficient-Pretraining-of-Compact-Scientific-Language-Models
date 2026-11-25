# PART3.md: Compact Transformer Training

**Project:** Tail-Aware Data Selection for Compute-Efficient Pretraining of Compact Scientific Language Models
**Stage:** Part 3, completed model training and validation
**Input:** Part 2 token files in `data/subsets/`

---

## 1. What this stage trains

The same compact decoder-only Transformer is trained on each of the five
Part 2 subsets:

| Subset | What it represents |
|---|---|
| `random` | Natural mix, no quality filter |
| `quality` | Quality filter, natural mix |
| `diversity` | Quality filter, balanced over text themes |
| `tail_aware` | Quality filter, tail-aware over arXiv fields (proposed) |
| `tail_only` | Tail-aware over fields, no quality filter (ablation) |

Every run gets the same tokenizer, model, optimizer, schedule, batch size,
context length, and token budget. Only `train.bin` changes.

**Completed comparison:** five seeds (42–46) for each subset, 25 runs total.
Each run uses a 60M-token budget, approximately two passes over its 30M-token
subset. All paired runs use the same number of optimizer steps.

`test.bin` is never opened in Part 3. It is reserved for Part 4.

---

## 2. Inputs

The committed Part 2 inputs are `tokenizer/tokenizer.json` and
`manifests/*.txt.gz`. The tokenized `.bin` files are git-ignored and must exist
locally:

```
data/subsets/
  random/train.bin      30.0M uint16 tokens
  quality/train.bin
  diversity/train.bin
  tail_aware/train.bin
  tail_only/train.bin
  val.bin                5.96M tokens, shared by every run
  test.bin               6.07M tokens, Part 4 only
  val_docs.parquet       per-document offsets and topic labels
  test_docs.parquet
  meta.json              vocab 16384, eot 0, pad 1
```

If those files are missing, rebuild them from the pinned source (about 25
minutes):

```bash
uv run python src/build_raw.py
uv run python src/clean_corpus.py
uv run python src/split_corpus.py
uv run python src/build_topics.py
uv run python src/export_tokens.py
```

If the Kaggle `vroomvroom2006/processedryan` parquet files are already in
`data/processed/`, skip the first three commands.

---

## 3. Model

See the [Transformer architecture](docs/transformer_architecture.md) for the
implemented model, the 17M and 34M configurations, and continuation behavior.

| Component | Value |
|---|---|
| Type | Decoder-only Transformer, pre-LayerNorm |
| Parameters | **17.1M** |
| Layers / heads / width | 6 / 6 / 384 |
| Context length | 512 tokens |
| Position encoding | Learned absolute embeddings |
| Attention | Causal scaled dot-product attention (`is_causal=True`) |
| MLP | GELU, 4x width |
| Embeddings | Input and output embeddings tied |
| Dropout | 0.0 |
| Vocabulary | 16,384 (fixed Part 2 tokenizer) |

Why this size: it is within the synopsis's 10-20M target, fits comfortably in
8 GB of VRAM, and leaves enough capacity for data-selection differences to
appear. A 17.1M model is intentionally small; this is a controlled data
experiment, not a attempt to maximize absolute language-model quality.

---

## 4. Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1, excluded from biases and LayerNorm |
| Peak learning rate | 6e-4 |
| Warmup | 2% of steps, linear |
| Decay | Cosine to 10% of peak |
| Batch size | 32 sequences x 512 tokens = 16,384 tokens/step |
| Gradient accumulation | 1 |
| Gradient clipping | 1.0 |
| Precision | bf16 autocast on CUDA, fp32 on CPU |
| Steps per run | 3,662 |
| Tokens per run | 59,998,208 (60M budget) |
| Seeds | 42, 43 |

The small difference between 60,000,000 and 59,998,208 is the whole-step
rounding; every subset has exactly the same number of steps.

---

## 5. Environment

The project uses Python 3.14 and a separate `train` dependency group, so the
Part 1/2 data environment does not need PyTorch:

```bash
uv sync --group train
uv run --group train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.14.1+cu130` and `True`. The training loop uses bf16 and PyTorch's
flash SDPA backend on the RTX 5050 Laptop (8.5 GB, compute capability 12.0).
No Unsloth dependency is used: Unsloth is a finetuning accelerator, while this
is a from-scratch model.

Measured on this machine: `torch 2.14.1+cu130`, CUDA 13.0, bf16 supported,
device `NVIDIA GeForce RTX 5050 Laptop GPU`, compute capability (12, 0).

---

## 6. Pilot

Run one short pilot before the ten-run schedule:

```bash
uv run --group train python src/train.py --subset random --seed 42 --max-steps 200 --out-dir runs/pilot/random_seed42 --eval-every 100
```

Pilot results:

| Metric | Value |
|---|---:|
| Device / precision | CUDA (RTX 5050 Laptop) / bfloat16 |
| Training-segment throughput | 55,626 tokens/s (58.9 s; includes the step-100 and step-200 validation passes) |
| Peak allocated VRAM | 4,703 MB |
| 200-step training loss | 5.84 (last logged 10-step mean) |
| 200-step validation loss | 5.8924 |
| 200-step validation perplexity | 362.26 |

The pilot confirms that the default batch size 32 with gradient accumulation 1
fits comfortably: peak memory is 4.7 GB of the 8.5 GB available. No gradient
accumulation or model-width change is needed. The pilot is not part of the
comparison and is stored under `runs/pilot/`.

---

## 7. Full run

From the project root, in a terminal you can leave open:

```bash
uv run --group train python src/train_all.py --seeds 42 43 44 45 46 --max-tokens 60000000
```

`train_all.py`:

- freezes every hyperparameter once at launch and passes it to each child, so
  a mid-schedule edit to `config.py` cannot mix settings across runs;
- skips a pair only if its `summary.json` is valid and covers the full token
  budget;
- resumes a partial pair from `ckpt_last.pt`, rejecting a resume whose
  training settings differ from the checkpoint;
- stops at the first failure instead of silently continuing;
- writes `runs/schedule.json` with every command and its status.

The runs are sequential and stream their logs to the terminal. The completed
25-run campaign logged approximately five hours of training and evaluation
on the RTX 5050 Laptop GPU.

If a run is interrupted, re-running the same command resumes it without
changing the data order or the learning-rate schedule, and keeps cumulative
wall time and peak-memory figures. `config_resume.json` records each resume.

To run one subset manually:

```bash
uv run --group train python src/train.py --subset tail_aware --seed 42
```

To resume an interrupted single run:

```bash
uv run --group train python src/train.py --subset tail_aware --seed 42 --resume
```

---

## 8. Outputs

Each run writes to `runs/<subset>/seed<seed>/`:

```
config.json       resolved model, data, optimizer, environment
run.log           human-readable progress
metrics.jsonl     step, tokens, train/val loss, perplexity, LR, grad norm,
                  tokens/s, peak memory
ckpt_last.pt      model + optimizer + scheduler, resumable
ckpt_best.pt      best-validation model only
model_last.pt     final model only (Part 4)
summary.json      final/best validation loss, perplexity, runtime, throughput
```

`runs/` is git-ignored. Checkpoints are large; keep them locally for Part 4.

After all runs, aggregate the comparison:

```bash
uv run --group train python src/plot_part3.py
```

That writes `reports/part3_report.json` and
`reports/figures/part3_val_loss.png`.

---

## 9. How to read the result

- The **primary number** is final validation loss at the fixed 60M-token budget,
  not the best checkpoint. Early stopping at different points would make the
  comparison unfair.
- `best_val_loss` is reported for analysis, but it is not the primary metric.
- Overall validation loss is dominated by head topics because the validation
  set is about 75% head. `tail_aware` deliberately halves the head, so its
  overall loss can look worse even if rare-topic loss improves. Part 4 must
  report head / mid / tail perplexity separately.
- `quality` and `random` share about 95% of their documents (Part 2). A small or
  null difference between them is expected and is itself a finding about this
  clean arXiv corpus.
- The five-seed review reports paired differences and variation across seeds.

---

## 10. Completed results

The original two-seed analysis was extended to five seeds (42–46) per strategy.
Each of the 25 runs processed 59,998,208 training tokens at the same model
size. The [five-seed review](reports/five_seed_results_review.md) contains the
paired analysis, and the [strategy summary](reports/5seed_strategy_summary.csv)
contains the measured values.

Tail-aware selection lowered mean tail loss from 3.9477 to 3.8571 relative to
random selection. Mean overall loss rose from 3.6514 to 3.6677. The tail benefit
and overall cost appeared in all five paired seeds. These are validation
results.

The [architecture document](docs/transformer_architecture.md) also covers the
34M random-only baseline. The [drafting and transfer guide](docs/phase1_drafting_application.md)
includes the matched SmolLM2 comparison and downstream drafting evaluation.
