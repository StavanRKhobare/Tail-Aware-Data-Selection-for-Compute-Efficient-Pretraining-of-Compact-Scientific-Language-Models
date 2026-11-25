# Phase 1 drafting application

Start from the project root with:

```powershell
uv run --group train --group transfer uvicorn demo.server:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765 and use Draft workspace.
The formatter works without checkpoints. The engine menu enables models as their local checkpoints become available. Use the `transfer` dependency group for SmolLM2 or an encoder-decoder checkpoint.

## Available controls

- 17M Transformer (V3), 17M and 34M rewrite pilots, SmolLM2 random and tail-aware rewrite pilots, an optional encoder-decoder transfer model, and the formatter. Automatic mode uses the V3 checkpoint when model wording is selected and falls back to the formatter if that checkpoint is absent.
- Compose from notes, preserve supplied sentences, or review model wording. Composition uses editing rules to combine compatible objectives and methods, then retains the supplied results and limitations.
- Preserve a working title or request a model suggestion. In source or composition mode, title suggestions use the separate title checkpoint when available.
- Optional rule-based editing converts a few imperative openings into prose. Exact duplicate sentences are removed.
- Output length, editable title and abstract, word count, copy, text export, and original output inspection.
- Numerical omission/addition and negation warnings are heuristics. An empty warning list does not establish factual correctness.

## Evaluation and limits

Twelve authored cases were tested through the actual API. Before source-preserving mode was added, 11 returned formatted drafts and one conflicting-input case returned an incomplete-format error. Rough notes caused repetition. A limitation was garbled, and a new-domain example omitted a 3% finding. Positive/negative reversals and supplied numbers were preserved in the initial sentence-based cases. Outputs are recorded in reports/draft_phase1_application_evaluation.json. These observations do not constitute a validated paraphrasing capability.

The workspace defaults to rule-based composition. The editor labels where its text came from. With supplied results locked, raw model wording appears separately and the editor shows the composed draft. The workspace flags model output that repeats most of the source sentences.

## Drafting datasets

| Pilot | Preparation | Purpose and limitations |
| --- | --- | --- |
| Reconstruction (V1) | Historical run used `prepare_draft_sft.py --limit 2000 --output data/draft_sft_v1` | Produced 1,031 train and 58 validation examples from the original training pool. It shuffled abstract sentences and supplied the title. The 512-token filter excluded 907 records, and four records were invalid. Retained examples averaged 405.17 tokens. |
| Structured source wording (V2/V3) | `src/prepare_draft_sft.py` and `src/prepare_draft_structured.py` | Synthetic reconstruction and extraction from the corpus train and validation splits. V3 includes prompts with missing titles or results. |
| Curated smoke pilot | `src/prepare_draft_curated.py` with local `data/draft_curated_v1/annotations.json` | Three train and three validation papers with assistant-authored rewrites. The script validates identity, split, context, and numerical support. Use this set for pipeline checks. |
| Rewrite comparison | `src/prepare_draft_rewrite.py --output data/draft_rewrite_v1` | 5,431 train and 117 validation examples. Original abstracts are the targets. The script shuffles sentences and removes some words to form synthetic notes. Exact duplicate abstracts are excluded across splits. The 512-token limit favors shorter abstracts, and synthetic shortening can affect meaning. |
| Title suggestions | `src/prepare_title_sft.py` | Separate supervision for proposing a title from an abstract. |

The current V2 preparer uses the corpus validation split. The original V1 split and counts describe the historical run. Generated datasets, curated annotations, and model weights are local files. The repository includes the authored comparison prompts and evaluation reports.

## Matched Phase 1 comparisons

The **Model comparisons** page shows one matched validation pair per model family at seed 42. The original 17M pair was trained from scratch and the SmolLM2 pair was continued from the same published base. Compare the tail-aware minus random loss differences within each family. Different tokenizers make absolute losses across the families unsuitable for ranking. The 34M checkpoint is a random-only drafting baseline pretrained on 120M tokens.

| Seed-42 validation comparison | Overall loss difference | Head | Mid | Tail |
| --- | ---: | ---: | ---: | ---: |
| 17M from scratch | +0.0032 | +0.0221 | -0.0447 | -0.1030 |
| SmolLM2 continued pretraining | +0.0122 | +0.0213 | -0.0127 | -0.0323 |

Differences are tail-aware minus random. Both comparisons show lower tail loss and higher head and overall loss with tail-aware selection. SmolLM2 used 59,998,208 training tokens per strategy. Its overall validation loss fell from 3.3422 at initialization to 2.8568 with random selection and 2.8691 with tail-aware selection. The adaptation comparison used one seed.

To reproduce the SmolLM2 comparison from the project root, use the processed corpus and topic metadata from the [README workflow](../README.md). Then fetch the pinned base and prepare its token streams:

```powershell
uv run --group train --group transfer hf download HuggingFaceTB/SmolLM2-135M --revision 93efa2f `
  config.json tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json model.safetensors `
  --local-dir runs/smollm2_base
uv run --group train --group transfer python src/smollm2.py prepare
uv run --group train --group transfer python src/smollm2.py train --subset random --out runs/smollm2/random/seed42
uv run --group train --group transfer python src/smollm2.py train --subset tail_aware --out runs/smollm2/tail_aware/seed42
uv run --group train --group transfer python src/smollm2.py evaluate `
  --random runs/smollm2/random/seed42/model_last.pt `
  --tail-aware runs/smollm2/tail_aware/seed42/model_last.pt `
  --out reports/smollm2_validation_comparison.json
```

The four rewrite pilots use 500 supervised steps, seed 42, a learning rate of 1e-5, and the same synthetic notes and original abstracts from `data/draft_rewrite_v1`. They start from the final 17M random, 34M random, SmolLM2 random, and SmolLM2 tail-aware checkpoints. Pass the relevant `--checkpoint` and a new `--output` to `src/train_draft_sft.py --data data/draft_rewrite_v1 --steps 500 --eval-every 500`. For either SmolLM2 checkpoint, also pass `--smollm2-base runs/smollm2_base`.

`python -m demo.compare_drafts --output reports/phase1_drafting_comparison.json` runs the same five authored cases through the draft API for the formatter, V3, and available pilots. The UI shows raw model wording and counts of successful API responses. Review the content against the notes even when a draft has valid formatting. Use a new output path when rerunning either report.

## Transfer-learning fallback

If the 34M attempt still copies rough notes or alters scientific claims, fine-tune a pretrained encoder-decoder model on the same rewrite split. Use each row's `prompt` as encoder input and its `target` as decoder output. Save the full Transformers model and tokenizer with `save_pretrained` into `runs/draft_transfer`, using SafeTensors for model weights. Install Transformers in the app environment and reload the workspace to enable the option. The app loads the exported files locally. The expected decoded output is the title text followed by `Abstract:` and the abstract text.

Both candidates should be compared with the current 17M model and formatter on the same held-out notes. Check that titles are complete, rough notes are meaningfully rewritten, results are correctly attributed, and quantities, negations, and limitations survive. Include cases with missing results. Phase 1 delivers the drafting workspace and a reproducible comparison setup. Any Hugging Face release should report the measured capability, failure cases, checkpoint provenance, and applicable licenses.
