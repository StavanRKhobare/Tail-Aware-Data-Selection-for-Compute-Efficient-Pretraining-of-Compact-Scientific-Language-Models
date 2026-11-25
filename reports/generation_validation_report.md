# Generation Validation Report

**Date:** 2026-11-10
**Scope:** generation only. No checkpoint, tokenizer, dataset, or training
configuration was modified.
**Code changed:** `src/generate.py` (rewritten), `tests/test_generate.py` (new).
**Checkpoints used:** read-only, `runs/<strategy>/seed<seed>/ckpt_best.pt`.

---

## 1. Root cause

The malformed output reported earlier ("... is" immediately followed by a
document title such as `Practical application in Chinese...`) had two
independent causes, both in the old `src/generate.py`.

### Defect 1 — out-of-distribution `[eot, eot]` prefix

The old `generate_from_checkpoint` encoded the prompt and then appended
**two** end-of-text ids before decoding:

```python
ids = tok.encode(prompt).ids + [eot, eot]
```

Training documents are `title\n\nabstract<|endoftext|>`, i.e. exactly **one**
EOT at a document boundary. An audit of every `train.bin` and `val.bin` file
found **zero** consecutive EOT pairs. `[eot, eot]` therefore never occurs in
training and is out of distribution. The model reacted to it as a signal to
start a new document, which is why ordinary prose prompts were answered with a
title.

Controlled evidence from the previous investigation: with the doubled EOT
prefix, 100/100 sampled generations began a new document at the boundary
(`boundary_joined = 1.0`); with no appended EOT, 0/100 did.

### Defect 2 — decoding prompt and continuation together

The old code decoded the entire id sequence in one call with
`skip_special_tokens=True`:

```python
text = tok.decode(prompt_ids + [eot, eot] + generated, skip_special_tokens=True)
```

EOT id `0` decodes to the empty string when skipped. The boundary therefore
disappeared, and the last prompt token was concatenated directly onto the first
generated token. That is the literal mechanism behind `"...method is"` +
`"Practical application..."` becoming `"...method isPractical application..."`.

This was a presentation bug on top of Defect 1. Even with a correct prefix,
decoding everything together would still hide where the prompt ended.

### What was *not* wrong

- Tokenizer: 16,384 vocab, EOT id 0, PAD id 1, round-trips verified.
- Corpus formatting and document boundaries: one EOT per document, no double
  EOTs, `title\n\nabstract` layout.
- Checkpoints: 10/10 load with `strict=True`, no missing/unexpected keys.

The generation defects explain the *fusion* and the *title-shaped* output. They
do **not** explain poor wording or repetition; those remain model-capability and
decoding limitations (Section 5).

---

## 2. Changes made

`src/generate.py` now exposes two explicit modes.

### Mode `continue` (text continuation)

- Encodes the prompt exactly as supplied.
- **Appends no EOT.**
- Reserves one context slot for the first generated token.
- Truncates long prompts from the left, keeping the most recent context.
- Stops on EOT if document-boundary stopping is enabled (it always is here),
  otherwise at `max_new_tokens`.

### Mode `document` (next-document generation)

- Requires an empty prompt.
- Conditions on **exactly one** EOT id, the boundary after a completed document.
- Stops when the model emits the next EOT.
- Prefix/suffix are returned such that concatenating them can never produce two
  EOTs (covered by a regression test).

### Shared guarantees

- EOT is detected by **token id**, never by decoded string.
- Prompt and continuation are decoded **separately**.
- Decoding uses `skip_special_tokens=False`; artificial special tokens are
  removed by id before decoding, so they cannot fuse text across a boundary.
- Empty continuation decodes to `""`, not to a special-token string.
- Generation settings (`temperature`, `top_k`, `top_p`, `decoding`, `seed`) live
  in `generate.py`, separate from training configuration.
- The default string return is preserved; `return_result=True` returns a
  `GenerationResult` with token-level metadata.
- `--mode` defaults to `document` for an empty prompt and `continue` otherwise.
- `--json` prints the full result, including stop reason and token ids.

New/changed public API:

```python
generate_from_checkpoint(checkpoint, prompt, *, mode="continue", ...)  # str
generate_from_checkpoint(..., return_result=True)                     # GenerationResult
generate_with_model(model, tok, prompt, *, mode, ...)                 # GenerationResult
build_generation_prefix(tok, prompt, *, mode, block_size)             # ids
```

`GenerationResult` records: `mode`, `prompt_text`, `continuation_text`, `text`,
`prompt_token_ids`, `prefix_token_ids`, `suffix_token_ids`,
`generated_token_ids`, `continuation_token_ids`, `eot_id`, `eos_position`,
`stop_reason`, `prompt_truncated`, `seed`, `temperature`, `decoding`,
`eos_generated`.

---

## 3. Test coverage

`tests/test_generate.py` (9 tests), all passing:

1. `test_continue_prefix_does_not_append_eot` — ordinary continuation adds no EOT.
2. `test_document_prefix_is_exactly_one_eot` — document mode starts with one EOT.
3. `test_generated_eot_terminates_document_generation` — EOT stops generation by id and is excluded from continuation text.
4. `test_prompt_and_continuation_are_returned_separately` — `text == prompt + continuation`.
5. `test_special_tokens_cannot_fuse_prompt_and_document` — a generated title cannot be glued to a prompt.
6. `test_empty_continue_prompt_is_rejected_clearly` — empty prompt raises `ValueError`.
7. `test_long_prompt_is_truncated_safely_from_the_left` — 16-slot context keeps 15 prompt tokens and no EOT.
8. `test_document_mode_rejects_a_nonempty_prompt` — mode contract enforced.
9. `test_prefix_and_suffix_never_contain_two_eots` — concatenation cannot rebuild `[eot, eot]`.

Full suite: **58 passed** (`uv run --group train pytest -q`).

The tests use a scripted model with deterministic next-token ids, so they test
the generation contract without depending on checkpoint quality.

---

## 4. Corrected generation examples

Settings: sampling, `temperature=1.0`, `seed=7`, 60 new tokens, no top-k/top-p.
Produced by the corrected code. `stop_reason` is `max_new_tokens` for all five
examples, i.e. no EOT occurred within 60 tokens; `eos_position` is `null`.

### 4.1 `quality/seed42`, `continue`

**Prompt**

> We propose a transformer model that

**Continuation**

> appropriately predicts by the end token feature of the model without modifying them to the input token items and guide text attribution. The key idea is to encode semantic contextual information into semantic feature vectors, represented by the BERT' layers and assign the most-hot vector translations to induce intrinsic to semantically similar instances, such

`prefix_token_ids == prompt_token_ids == [1324, 643, 260, 2761, 343, 341]`,
`suffix_token_ids == []`, `eot_id == 0`.

The prompt is not repeated, no title appears, and no EOT was injected.

### 4.2 `quality/seed42`, `document`

**Continuation**

> Practical application in Chinese: English Economonized Models for Patient Management in German Clinical Auditacu & QA
>
> Inth establishes EffectiveLLM-based LLM-based chatbots for job planning tailored to online and online conversations, accurate assessments are essential to induce health risks. While Large Language Models

Structure is now a title, a real blank-line boundary, and abstract-like prose.
The title is generic and some words are malformed, but the text is not fused to
any prompt.

### 4.3 `tail_aware/seed42`, `continue`

**Prompt**

> The main contribution of this paper is

**Continuation**

> a special application in statistics: feature selection that is often limited to one or less available items andMAP use only a few labeled items, unless a fair score shares a specific feature characteristic, a sequential clustering algorithm such that it does not have access to pre-compose the user. To improve the performance

Continuation follows the prompt grammatically. `andMAP` shows the model's own
spacing error, not a boundary artifact.

### 4.4 `tail_aware/seed42`, `document`

**Continuation**

> Practical application in Chinese language data analytics
>
> Evolutionary approaches to sequential decision-making and guide use can help alleviate the burden of global insights. Although neural networks exist for the efficient exploration of action tailored to tasks and methods for generating accurate results, they lack theoretical guarantees. To overcome this, several

Clean title → `\n\n` → abstract structure.

### 4.5 `random/seed43`, `document`

**Continuation**

> Peslished Chinese: English Korean ClarEvolution Better Cocking artefacts from The Auditory for Low-Resource Low-Resource Scenarios
>
> LLM agents are upcoming the growing access job of large language models and tools for generating accurate, accurate, and socially-aligned language models (S compress

Same structural pattern, with clear lexical degeneration (`Peslished`,
`Cocking`, repeated `accurate`). This is a model-quality signal.

### Before/after contrast

| | Old code | Corrected code |
|---|---|---|
| Prompt-conditioned output | prompt fused with a title | prompt preserved, continuation separate |
| Appended boundary | `[eot, eot]` | none (`continue`) / one EOT (`document`) |
| EOT detection | string removal via `skip_special_tokens` | token id |
| Decoding | prompt + specials + continuation together | prompt and continuation separately |
| Recorded stop reason | none | `eos` or `max_new_tokens`, with `eos_position` |

---

## 5. What remains a model limitation

These are **not** generation bugs and must not be presented as such:

1. **Repetition under greedy decoding.** Deterministic decoding is much more
   repetitive than sampling. Across the ten checkpoints, greedy 3-gram repeat
   fractions were 0.37–0.65 and maximum 4-gram counts 12–121. Sampling reduces
   this substantially (3-gram repeat fraction roughly 0.02–0.07 in the
   fixed-prompt diagnostic).
2. **Generic or malformed titles.** Document mode reliably produces a title-like
   first line, but the titles are templated (`Practical application in ...`) and
   occasionally contain malformed words. The models are undertrained: 60M tokens
   for 17.1M parameters is about 3.5 tokens/parameter, versus roughly 20 in
   Chinchilla scaling, and validation loss was still falling at the final step.
3. **Occasional long repeats.** Even with sampling, individual generations can
   contain repeated token runs; this is checkpoint quality, not prompt handling.

The corrected code makes these limitations **visible and measurable** instead of
disguising them as boundary fusion. Qualitative examples from the old API should
not be reused as report evidence.

---

## 6. Verdict

- **Confirmed:** the malformed prompt/title join was caused by the old
  generation code, specifically the doubled out-of-distribution EOT prefix and
  whole-sequence decoding with special-token skipping.
- **Confirmed fixed:** two explicit modes, single-EOT document start, token-id
  EOS detection, separate decoding, and regression tests.
- **Not implicated:** tokenizer, corpus formatting, document boundaries,
  checkpoints, and training configuration.
- **Remaining:** poor fluency and repetition are properties of the current
  checkpoints. They should be reported as capability limitations and addressed
  by more training or a larger model, not by further generation patching.

---

## 7. Reproduction

```powershell
uv run --group train pytest -q
uv run --group train python src/generate.py `
  --checkpoint runs/quality/seed42/ckpt_best.pt `
  --mode document --max-tokens 60 --seed 7 --json
uv run --group train python src/generate.py `
  --checkpoint runs/quality/seed42/ckpt_best.pt `
  --prompt "We propose a transformer model that" `
  --max-tokens 60 --seed 7
```
