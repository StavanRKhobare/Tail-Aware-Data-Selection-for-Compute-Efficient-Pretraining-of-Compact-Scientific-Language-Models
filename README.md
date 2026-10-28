# Tail-Aware Data Selection for Compact Scientific LMs

Part 1: corpus collection. Source: librarian-bots/arxiv-metadata-snapshot
(pinned revision in data/raw/source_info.json). Details: DATASET.md.

Run order:
  uv run python src/build_raw.py
  uv run python src/inspect_corpus.py
  uv run python src/clean_corpus.py
  uv run python src/split_corpus.py

The Part 1 outputs (data/processed/*.parquet) are also on Kaggle:
vroomvroom2006/processedryan. Download them into data/processed/ to skip Part 1.

Part 2: data selection and tokenization. Details: PART2.md.

Run order (about 20 min on 4 CPU cores, no GPU needed):
  uv run python src/train_tokenizer.py   # BPE tokenizer + token counts
  uv run python src/score_quality.py     # quality, near-duplicates, leakage
  uv run python src/build_topics.py      # head / mid / tail + text clusters
  uv run python src/select_subsets.py    # the 5 training subsets (30M tokens each)
  uv run python src/export_tokens.py     # .bin token files for Part 3
  uv run pytest                          # unit tests

Committed Part 2 outputs: tokenizer/tokenizer.json, manifests/*.txt.gz
(the document ids of every subset), reports/*.json, reports/figures/.
