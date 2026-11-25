# Tail-Aware Data Selection for Compact Scientific LMs

This project tests how the mix of scientific papers affects next-token prediction under a fixed training budget. It includes a compact Transformer trained from scratch, a SmolLM2 continued-pretraining comparison, and the local TailLab demo.

## Results

Across five seeds at approximately 60M tokens per run, tail-aware selection lowered the 17M model's mean rare-topic validation loss from 3.9477 to 3.8571. Its mean overall validation loss rose from 3.6514 to 3.6677. A seed-42 comparison with SmolLM2 showed the same direction. The [five-seed review](reports/five_seed_results_review.md) and [drafting guide](docs/phase1_drafting_application.md) document the results and their limits.

## Run TailLab

```bash
uv run --group train --group transfer uvicorn demo.server:app --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. The app can format notes and display committed reports before you have model checkpoints. The [drafting guide](docs/phase1_drafting_application.md) lists the experimental engines and outlines how to reproduce the comparisons.

## Reproduce the experiment

The pinned arXiv corpus and preprocessing steps are described in [DATASET.md](DATASET.md). The processed Parquet files are also available as `vroomvroom2006/processedryan` on Kaggle.

```bash
uv run python build_corpus.py
uv run python src/score_quality.py
uv run python src/build_topics.py
uv run python src/export_tokens.py
```

The fixed tokenizer and five subset manifests are already in the repository. See [PART2.md](PART2.md) to rebuild the tokenizer or manifests. See [PART3.md](PART3.md) for training, checkpoints, and interpretation, and [Transformer architecture](docs/transformer_architecture.md) for the 17M and 34M models trained from scratch.

```bash
uv run --group train python src/train_all.py --seeds 42 43 44 45 46 --max-tokens 60000000
uv run --group train python audit_results.py --seeds 42 43 44 45 46 --tag 5seed
uv run --group train pytest -q
```

Corpus exports, generated drafting datasets, and checkpoints are kept locally. The repository includes the tokenizer, subset manifests, authored evaluation cases, and reports. Tests that inspect local drafting data skip when those files are absent.
