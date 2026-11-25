# Investigation: Generation Quality and Experimental Validity

> **STATUS (2026-11-10): both generation defects identified below are fixed.**
> `src/generate.py` now exposes explicit `continue` and `document` modes, appends
> no doubled EOT, detects EOS by token id, and decodes prompt and continuation
> separately. Regression coverage is in `tests/test_generate.py`; corrected
> examples and test details are in `reports/generation_validation_report.md`.
> Statements below such as "not modified" or "until fixed" describe the state at
> the time of the investigation and are retained as historical evidence.

**Scope:** Parts 1–3 of *Tail-Aware Data Selection for Compute-Efficient Pretraining of Compact Scientific Language Models*.

**Constraint compliance:** no retraining, no checkpoint overwrite, no corpus/tokenizer/preprocessing change, no training-pipeline change. `src/generate.py` was inspected but **not modified at the time of this investigation**. Only diagnostic code was replaced/added.

**Reproduce:**

```powershell
uv run --group train python investigate_pipeline.py
uv run --group train python audit_training.py
uv run --group train python probe_eot.py
uv run --group train python deep_analysis.py --seeds 42 43 --sample-seeds 0 1
uv run --group train python deep_analysis.py --seeds 42 --decoding greedy
uv run --group train python eval_by_group.py
uv run --group train pytest -q
```

All diagnostic scripts are read-only. Outputs are in `reports/`.

---

## A. Findings

### A1. `src/generate.py` appends two EOT tokens to every prompt — High

**File/function:** `src/generate.py`, `generate_from_checkpoint`, line 68.

```python
encoded = tok.encode(prompt)
ids: list[int] = encoded.ids[: cfg.block_size - 2] + [eot, eot]
```

**Evidence:**

- The training format is `title\n\nabstract<|endoftext|>` with **exactly one** EOT per document (`src/export_tokens.py:25-31`).
- There are **zero occurrences of two consecutive EOT ids** in any training file or in `val.bin`:

```
data/subsets/val.bin            double_EOT=0  tokens=5,989,130
data/subsets/diversity/train.bin double_EOT=0  tokens=29,999,698
data/subsets/quality/train.bin   double_EOT=0  tokens=29,999,935
data/subsets/random/train.bin    double_EOT=0  tokens=29,999,918
data/subsets/tail_aware/train.bin double_EOT=0 tokens=29,999,918
data/subsets/tail_only/train.bin  double_EOT=0 tokens=29,999,854
```

- `[eot, eot]` is therefore **out-of-distribution**. Even one EOT means “a new document starts next”; two EOTs are a context the model has never seen.
- The model has learned this meaning: at document-final target positions it assigns `p(EOT) = 0.510–0.566`, with EOT recall `0.834–0.876` and false-positive rate `0.0034–0.0041` (128 fixed validation windows; `probe_eot.py`). At all other positions `p(EOT) ≈ 0.0019–0.0022`.

**Why it is problematic:** after `[eot, eot]` the model does what the corpus taught it to do at a document boundary: it emits a **new paper title**. This is not a model failure; it is a prompt-construction mismatch. It also explains why the behavior is identical across all five checkpoints.

**Severity:** High for generation/analysis. Not Critical: training and validation loss are unaffected.

---

### A2. `src/generate.py` decodes prompt and continuation in one call with special tokens skipped — High

**File/function:** `src/generate.py`, `generate_from_checkpoint`, lines 96–105.

```python
all_ids = idx.squeeze().tolist()
while all_ids and all_ids[-1] == eot:
    all_ids.pop()

tokens = [tok.decode([i]) for i in all_ids]   # dead code, unused
text = tok.decode(all_ids).strip()
```

`Tokenizer.decode` defaults to `skip_special_tokens=True`.

**Evidence (same model/prompt/seed, `tail_aware/seed42`, prompt `"The key innovation in machine learning is"`):**

```
prefix ids                      : [1351, 1355, 7523, 288, 1008, 464, 351, 0, 0]
decode(prefix, keep specials)   : 'The key innovation in machine learning is<|endoftext|><|endoftext|>'
decode(prefix + continuation)   : 'The key innovation in machine learning isDenseNet: An efficient and robust kernel of X-ray imaging'
decode(prompt, keep specials)   : '...is<|endoftext|><|endoftext|>'
tok.decode([0])                 : ''
tok.decode([0], keep=True)      : '<|endoftext|>'
```

The EOT ids are **in the middle** of `all_ids`, not at the end, so the `while ... pop()` loop does not remove them. `tok.decode(..., skip_special_tokens=True)` then deletes them, fusing the prompt’s last character with the first character of the generated title.

**Why it is problematic:** the returned string is not the text the model actually produced. It hides the document boundary and makes `is` + `DenseNet` look like one malformed word. The apparent “missing space” is a **decoding artifact of the artificial EOT tokens**, not a tokenizer whitespace bug and not a BPE issue.

**Severity:** High. This makes every `src/generate.py` output unsuitable as report evidence until fixed.

---

### A3. The previous `deep_analysis.py` used a different prompt per model — High (for the analysis)

**Old behaviour:** each of the five models received a different prompt (`tail_aware` → `"The key innovation..."`, `quality` → `"Recent advances..."`, etc.), with no token ids, no EOS-by-id detection, no stop reason, no length/repetition metrics, and no structured output.

**Why it is problematic:** model outputs cannot be compared. Any qualitative claim such as “model X writes better titles” is confounded by prompt choice. The old script also checked the decoded string for `"<|endoftext|>"`, which never appears because special tokens are skipped — so its EOS check was vacuous.

**Status:** replaced by the new `deep_analysis.py` (Section D).

**Severity:** High for experimental claims; now fixed.

---

### A4. EOT is a document separator, but generation treats it as an end-of-sequence token — Medium

**File/function:** `src/generate.py:91` (`if idx_next.item() == eot: break`); training (`src/export_tokens.py:29`, `src/data.py:28-33`) uses EOT as a separator and **continues** training across it.

**Evidence:** EOT is appended once per document and is included in targets. The model is trained to predict the next document’s first token after EOT. Generation stops at the first EOT instead.

**Why it is problematic:** it is a semantic mismatch. Stopping is reasonable if the goal is “generate exactly one document”, but it silently truncates a continuation at the first document boundary. It does **not** cause the title-after-prompt symptom (in the `eot2` diagnostic runs, only 10–30% of generations hit EOS within 160 tokens), but it should be made explicit.

**Severity:** Medium (design/semantics, not a crash).

---

### A5. `src/generate.py` exposes only temperature; no top-k/top-p/repetition penalty — Medium

**File/function:** `src/generate.py:79-88`.

```python
next_logits = logits[:, -1, :]
if temperature != 1.0:
    next_logits = next_logits / temperature
prob = torch.softmax(next_logits, dim=-1)
idx_next = torch.multinomial(prob, num_samples=1)
```

No top-k, top-p, repetition penalty, or no-repeat n-gram constraint. With a 16,384-token vocabulary, rare subword tokens with probability `~1e-4` can be sampled, producing invented words. The new diagnostic supports top-k/top-p/greedy, but `src/generate.py` itself does not.

**Why it is problematic:** it amplifies malformed words and makes qualitative output unnecessarily unstable. Not a correctness bug; a missing control.

**Severity:** Medium.

---

### A6. Training windows cross document boundaries without attention reset — Low/Medium

**File/function:** `src/data.py:24-33` (non-overlapping 512-token windows, shifted targets); `src/model.py:119-135` (no document mask).

**Evidence:** 11,695 of 11,697 validation windows (99.98%) contain at least one internal EOT; mean internal EOTs per window = 2.08. Targets are not masked at boundaries.

**Why it matters:** this is standard nanoGPT-style packing and is not a bug. It does mean the model can attend across documents and is explicitly trained to predict the next document after EOT. That is exactly why appending EOT to a free-form prompt switches the model into “new paper” mode. It can also blend unrelated documents, but no evidence shows this is the cause of the observed formatting artifacts.

**Severity:** Low/Medium (design choice; document it).

---

### A7. Tokenizer decoder configuration disagrees with the pre-tokenizer — Low

**File/function:** `src/train_tokenizer.py:30-32`.

```python
tok.normalizer = normalizers.NFC()
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()      # default add_prefix_space=True
```

`tokenizer.json` confirms `pre_tokenizer.add_prefix_space = false` and `decoder.add_prefix_space = true`.

**Why it matters:** inconsistent configuration. In practice, round-trips are exact, including leading spaces, double spaces, newlines, Unicode, and LaTeX, so this is **not** the cause of the observed artifacts.

**Severity:** Low (cosmetic/consistency).

---

### A8. Minor issues in `src/generate.py` — Low

- `src/generate.py:103`: `tokens = [...]` is computed and never used (dead code).
- `src/generate.py:99-100`: the trailing-EOT `while` loop cannot remove the artificial EOTs, which are mid-sequence. Misleading comment.
- `src/generate.py:105`: `.strip()` removes leading/trailing whitespace from the returned text, which can delete intentional formatting.
- `src/generate.py:63`: tokenizer path is hard-coded relative to the current working directory.
- No RNG seed is set, so sampling is not reproducible across runs.
- `src/generate.py:68`: prompt truncation can cut mid-token.

**Severity:** Low.

---

### A9. `src/train_tokenizer.py` reads the test split for token counting — informational, not leakage

**File/function:** `src/train_tokenizer.py:63-66, 96-100`.

The tokenizer is trained **only** on `train["text"]` (line 85). `test.parquet` is read afterwards solely to count tokens and report fertility. No test text influences the BPE merges. This is consistent with the reported token counts and is not a leak.

**Severity:** Informational / none.

---

### A10. What is working correctly (explicitly verified)

| Check | Evidence | Verdict |
|---|---|---|
| Tokenizer vocabulary | 16,384 tokens; EOT id 0; PAD id 1; both are `added_tokens` with `special: true`; no BOS/UNK (byte-level BPE covers every byte) | Correct |
| Tokenizer round-trip | 5/5 representative scientific/Unicode/LaTeX samples exact; `tests/test_tokenizer.py` passes | Correct |
| Whitespace/newlines | `"a b"`, `"a  b"`, `"a\n\nb"`, leading spaces all reconstruct exactly | Correct |
| Corpus format | 2,000/2,000 sampled rows contain exactly one `"\n\n"`; 0 tabs; 0 double spaces; no literal EOT string | Correct |
| Document boundaries | First 500 val docs: 500/500 end in EOT; 500/500 next id is not EOT; 500/500 have no internal EOT | Correct |
| EOT count | `val.bin` has 5,989,130 tokens = 5,964,729 text tokens + 24,401 EOT | Consistent |
| Train token count | 108,100,026 text tokens + 440,584 EOT = 108,540,610 | Consistent |
| Checkpoint loading | 10/10 checkpoints load with `strict=True`; no missing/unexpected keys; 17.136M params each | Correct |
| Checkpoint config | All ten checkpoints: vocab 16,384; block 512; 6 layers; 6 heads; width 384; dropout 0.0; bias True | Identical |
| Training configs | All ten runs identical except `subset`/`seed`; no mismatches | Correct |
| Loss calculation/eval | `F.cross_entropy` over all shifted targets; token-weighted validation accumulation | Correct |
| Part 3 report | Recomputed means match `reports/part3_report.json` to `<5e-7` | Correct |
| Test split | No training/eval code opens `test.bin`; only tokenizer counting reads `test.parquet` | No leakage |

**Severity:** None — these are working as intended.

---

### A11. Training quality (per-run evidence)

All ten runs: 3,662 steps, 59,998,208 tokens, 17.136M params.

| Subset | seed42 final val | seed43 final val | mean | best_step |
|---|---:|---:|---:|---:|
| quality | 3.6467 | 3.6464 | 3.6465 | 3662 |
| random | 3.6570 | 3.6598 | 3.6584 | 3662 |
| tail_aware | 3.6602 | 3.6642 | 3.6622 | 3662 |
| diversity | 3.6668 | 3.6601 | 3.6634 | 3662 |
| tail_only | 3.6630 | 3.6800 | 3.6715 | 3662 |

- Step-0 validation loss is identical across subsets for a given seed (`9.7566` for seed42, `9.7788` for seed43), confirming the same initialization per seed.
- Validation loss is still decreasing at the final step for every run; `best_step == 3662` for all ten. No overfitting.
- Gradient norms: min 0.44, mean ~0.78, max 2.05. No instability or loss spikes.
- Train loss final ~3.58–3.71; val loss ~3.65. Small gap, no overfitting.
- Final perplexity ~38.5.

**Verdict:** stable, reproducible, and not overfitting. Undertrained relative to compute-optimal scaling (Section B4).

---

## B. Root-cause assessment

### B1. Confirmed bugs / genuine defects

1. **Prompt construction puts the model into document-start mode.**
   `src/generate.py:68` appends `[eot, eot]`, an out-of-distribution context (zero double-EOTs exist in training). The model therefore emits a title. This is the direct cause of “title immediately after the prompt”.

2. **Prompt/continuation fusion during decoding.**
   `src/generate.py:105` decodes prompt + artificial EOTs + continuation in one call with `skip_special_tokens=True`. The EOTs vanish, so `...is` and `DenseNet...` are concatenated with no separator. This is the direct cause of `isLearning...` / `isHausally...` / `isDenseNet...`.

3. **The old `deep_analysis.py` was not a fair comparison.**
   One different prompt per model; no token-level EOS detection; no structured output. Its qualitative conclusions were not usable.

4. **EOT is trained as a separator but used as a stop token.**
   Semantics are inconsistent; generation may truncate at a document boundary. Not the cause of the title artifact, but a correctness/documentation issue.

### B2. Plausible explanations (not confirmed bugs)

- **Malformed words** (`Faught`, `foritigate`, `Daining`, `Thelicit`, `theTheoretic`, `Marginal-Sational`, `M4Gwhuhuhu`): the tokenizer is lossless; these are generated subword sequences. A 17M model trained on 60M tokens has a broad, noisy next-token distribution; full-vocabulary temperature sampling can pick rare fragments. This is a model-capacity/sampling limitation, not tokenizer corruption.
- **Repetition**: model-level degeneration. Greedy decoding is much worse (Section D); sampling with no repetition penalty occasionally collapses (one 127-token `"H"` run). Not a training-loop bug.
- **Titles even without EOT**: the model is a document LM, not an instruction-tuned assistant. A prompt that looks like a title or a sentence fragment can still be continued with title/abstract-like text. This is expected for the task and data.

### B3. Expected limitations of a compact model

- 17.1M parameters, 60M training tokens, no instruction tuning, no RLHF, no repetition penalty.
- Validation perplexity ~38.5; absolute text quality will be low.
- The model’s EOT probability at true boundaries is ~0.52 (recall ~0.85), not ~1.0. That is good for its size but not perfect.

### B4. Undertraining assessment

Chinchilla-style compute-optimal training would use roughly `20 × 17.1M ≈ 343M` tokens. The runs used 60M tokens (`≈3.5` tokens/parameter, ~5.7× short). Validation loss is still falling at step 3,662. So the models are **undertrained in absolute terms**, but the comparison is a valid **fixed-budget** comparison. Undertraining does not explain the prompt/title fusion; that is deterministic and disappears with `--prompt-suffix none`.

---

## C. Experimental validity

### C1. What is fair

- Same tokenizer file for all runs (`tokenizer/tokenizer.json`, vocab 16,384; EOT 0; PAD 1).
- Same architecture and hyperparameters for all ten runs (verified field-by-field; zero mismatches).
- Same token budget: 59,998,208 tokens per run (3,662 × 512 × 32).
- Same validation file and token-weighted loss calculation for all runs.
- Same initialization per seed: step-0 validation loss is identical across subsets for seed 42 (`9.7566`) and seed 43 (`9.7788`).
- Seed 42/43 are intentional replications: same `train.bin` per subset, different initialization/shuffle order.
- `ckpt_best.pt` equals the final weights in every run (`best_step = 3662`), so there is no checkpoint-selection bias.
- Test split is untouched by training and validation.

**Conclusion:** the training mechanics are fair and reproducible.

### C2. What limits the strength of the comparison

1. **Only two training seeds.** Differences of 0.003–0.012 nats are close to seed noise. No confidence intervals or significance tests were pre-registered.
2. **`quality` and `random` share 94.8% of documents** (Jaccard). Their difference is real in both seeds (quality better by ~0.012 nats), but it comes from a narrow ~5% document difference.
3. **`tail_aware` and `tail_only` share 94.6% of documents.** They are not independent strategies.
4. **Overall validation loss is dominated by the head.** In `val.bin`, the head/mid/tail token counts are 4,525,528 / 1,196,923 / 266,413 (75.6% / 20.0% / 4.45%). A large tail improvement can be hidden by a small head regression.
5. **Overall loss is not a per-group test.** `tail_aware` is designed to shift the distribution; scoring only the natural mix penalizes exactly the trade-off it makes.

### C3. Per-group validation loss (new diagnostic)

Token-level validation loss by group, all 11,697 val windows, same attribution for every model:

| Model | seed | overall | head | mid | tail |
|---|---:|---:|---:|---:|---:|
| random | 42 | 3.6570 | 3.6176 | 3.7402 | 3.9535 |
| random | 43 | 3.6598 | 3.6204 | 3.7427 | 3.9572 |
| quality | 42 | 3.6467 | 3.6073 | 3.7294 | 3.9440 |
| quality | 43 | 3.6464 | 3.6070 | 3.7296 | 3.9423 |
| diversity | 42 | 3.6668 | 3.6254 | 3.7594 | 3.9541 |
| diversity | 43 | 3.6601 | 3.6189 | 3.7527 | 3.9440 |
| tail_aware | 42 | 3.6602 | 3.6397 | 3.6955 | 3.8505 |
| tail_aware | 43 | 3.6642 | 3.6437 | 3.6998 | 3.8523 |
| tail_only | 42 | 3.6631 | 3.6426 | 3.6986 | 3.8511 |
| tail_only | 43 | 3.6800 | 3.6594 | 3.7169 | 3.8647 |

Full-validation perplexities (mean over seeds, `exp(loss)`): **quality 38.34**, **random 38.80**, **tail_aware 38.95**, **diversity 38.99**, **tail_only 39.31**. Per-group tail perplexity: **random ≈ 52.2**, **quality ≈ 51.6**, **tail_aware ≈ 47.1**. The fixed 512-window diagnostic (`fixVal`) reproduces the same ranking as the full validation loss.

**Interpretation:**

- `quality` is better than `random` in **every** group and both seeds. This is the most robust result.
- `tail_aware` improves **tail** loss by ~0.104 nats (≈9.9% lower tail perplexity) and **mid** loss by ~0.045 nats, but worsens **head** loss by ~0.023 nats. Because head is 75.6% of tokens, overall loss is slightly worse (+0.003 to +0.004 nats).
- `tail_only` behaves similarly but is worse than `tail_aware` on head and overall (it uses the eligible pool, including low-quality documents).
- `diversity` is not better than `random` overall in either seed.

**This changes the scientific conclusion:** `tail_aware` is not a win on the pre-registered overall validation loss. It is a distributional trade-off that helps mid/tail at the cost of head. Whether that is desirable depends on the target deployment distribution and must be stated explicitly.

### C4. Generation comparison

The old generation outputs are invalid for cross-model comparison because of different prompts and the EOT/fusion artifacts. The new diagnostic fixes both. However, qualitative generation should still not be used to rank strategies: 5 prompts × 2 sampling seeds is illustrative, not a benchmark. Use aggregate validation loss and per-group loss for ranking; use generation only for clearly labelled qualitative examples.

---

## D. Diagnostic improvements

### D1. New `deep_analysis.py`

Read-only, same fixed prompt set and generation settings for every checkpoint.

Features:

1. Loads all five checkpoints (and both training seeds by default in the report run); `GPTConfig(**ckpt["config"])`; `load_state_dict(..., strict=True)`.
2. Fixed prompt set:
   - `"The key innovation in machine learning is"`
   - `"Recent advances in large language models"`
   - `"Transformers have demonstrated"`
   - `"We propose a method for"`
   - `"Rare events in scientific texts"`
3. Identical decoding settings for every model; configurable `--temperature`, `--top-k`, `--top-p`, `--max-new-tokens`, `--decoding {sample,greedy}`, `--sample-seeds`.
4. Prompt construction is explicit and configurable: `--prompt-suffix {eot2,eot1,none,newline}`. `eot2` reproduces the current `src/generate.py`; `none` is the clean continuation baseline.
5. Records prompt text and continuation **separately**, plus `prompt_token_ids`, `prefix_token_ids`, `generated_token_ids`, `continuation_token_ids`, `eos_id`, `eos_position`.
6. Detects EOS by token id, never by string search. Reports `stop_reason` and `eos_generated`.
7. Reports output length in tokens and characters.
8. Repetition metrics: token 2/3/4-gram distinct ratio and repeat fraction, max n-gram count, word 3/4-gram repeat fraction, longest same-token run, word type-token ratio. These are descriptive, not “coherence scores”.
9. Records a `boundary_joined` flag and `prompt_plus_continuation_equals_full_skip` to make the fusion bug measurable.
10. Computes a fixed, evenly spaced validation loss on 512 windows for every checkpoint (sanity check against the full validation loss).
11. Writes `reports/deep_analysis.json`, `reports/deep_analysis.csv`, `reports/deep_analysis_summary.csv`.
12. Supports deterministic decoding (`--decoding greedy`) alongside temperature sampling.

### D2. Example output (final run: 5 models × 2 training seeds × 5 prompts × 2 prompt modes × 2 sampling seeds = 200 generations)

```
model       suffix    n   eos%  joined%    tok  tok3rep  tok4rep  4gmax   ckptVal    fixVal
-------------------------------------------------------------------------------------------
random      eot2     20   25.0    100.0  157.2    0.022    0.008      4    3.6584    3.6640
random      none     20   55.0      0.0  122.5    0.037    0.017      5    3.6584    3.6640
quality     eot2     20   30.0    100.0  156.0    0.039    0.016      4    3.6465    3.6544
quality     none     20   35.0      0.0  141.9    0.034    0.013      6    3.6465    3.6544
diversity   eot2     20   20.0    100.0  155.4    0.041    0.021      5    3.6634    3.6714
diversity   none     20   55.0      0.0  122.3    0.075    0.056    124    3.6634    3.6714
tail_aware  eot2     20   10.0    100.0  157.5    0.044    0.020      6    3.6622    3.6661
tail_aware  none     20   55.0      0.0  134.3    0.032    0.014      4    3.6622    3.6661
tail_only   eot2     20   15.0    100.0  158.8    0.027    0.010      4    3.6715    3.6744
tail_only   none     20   60.0      0.0  106.8    0.038    0.018      7    3.6715    3.6744
```

Key result: `boundary_joined = 100%` for every `eot2` generation and `0%` for every `none` generation. The fusion is caused by prompt construction, not by the tokenizer or the checkpoint.

### D3. Same model/prompt/seed, `eot2` vs `none`

```
--- eot2 ---
prompt_text           : 'The key innovation in machine learning is'
prompt_suffix_ids     : [0, 0]
first generated pieces: ['D', 'ense', 'Net', ':', ' An', ' efficient', ' and', ' robust', ...]
continuation          : 'DenseNet: An efficient and robust kernel of X-ray imaging\n\nThis article proposes...'
full skip-special     : 'The key innovation in machine learning isDenseNet: An efficient...'
full keep-special     : 'The key innovation in machine learning is<|endoftext|><|endoftext|>DenseNet: ...'
boundary_joined       : True
stop_reason           : max_new_tokens

--- none ---
prompt_text           : 'The key innovation in machine learning is'
prompt_suffix_ids     : []
first generated pieces: [' the', 'Theoretic', ' Equ', 'ity', ' of', ' M', 'arginal', '-', ...]
continuation          : ' theTheoretic Equity of Marginal-Sational Integrity (PSOs), holds ...'
full skip-special     : 'The key innovation in machine learning is theTheoretic Equity ...'
boundary_joined       : False
stop_reason           : eos (position 155)
```

With `none`, the model produces a leading-space token and continues; it is still a weak model, but the prompt/title boundary is no longer corrupted by skipped EOTs.

### D4. Greedy decoding demonstrates model-level degeneration

`reports/deep_analysis_greedy.*` (5 models × 5 prompts × 2 modes):

```
random      eot2   3gram_repeat=0.510  4gram_max=38
random      none   3gram_repeat=0.594  4gram_max=90
quality     eot2   3gram_repeat=0.460  4gram_max=12
quality     none   3gram_repeat=0.369  4gram_max=13
diversity   eot2   3gram_repeat=0.571  4gram_max=51
tail_aware  none   3gram_repeat=0.557  4gram_max=121
```

Examples:

```
tail_aware, none: "... predict the next-generation of a target target target target ..."
quality, none:    "... a neural network that learns a neural network that learns to learn a neural network ..."
```

Greedy decoding is far more repetitive than temperature sampling (`tok3rep` 0.37–0.65 vs 0.02–0.08). This confirms repetition is a property of the compact model/decoding, not a tokenizer or data-pipeline defect.

### D5. Other diagnostic artifacts

- `reports/val_loss_by_group.json` — per-group validation loss (Section C3).
- `investigate_pipeline.py` — tokenizer round-trips, corpus formatting, EOT boundary audit, window audit, checkpoint restoration, token-level generation trace.
- `audit_training.py` — config identity, loss curves, gradient norms, seed pairing, report consistency.
- `probe_eot.py` — EOT probability/recall/false-positive probe.
- `eval_by_group.py` — token-level per-group validation loss.

All 49 repository tests still pass (`uv run --group train pytest -q`).

---

## E. Recommended next steps

### 1. Must fix before trusting/using generated text

1. **Fix prompt construction and decoding in `src/generate.py`.** Proposed minimal change (not applied):
   - Add an explicit mode:
     - `continue`: no EOT suffix; decode prompt and continuation separately; return `prompt_text + continuation_text`.
     - `document`: one EOT suffix (not two); return `prompt_text + "\n\n" + continuation_text` (or another explicit separator).
   - Never decode artificial EOTs inside the returned string. Decode `prompt_ids` and `generated_ids` separately, or replace EOT ids with `"\n\n"` before decoding.
   - Record `stop_reason` and `eos_position`; do not silently truncate.
   - Set and record an RNG seed.
2. **Re-run all qualitative examples** with the fixed construction. Do not use the existing `isLearning...` / `isHausally...` strings in the report.
3. **Do not claim `tail_aware` beats `random` on overall validation loss.** It does not: +0.003 to +0.004 nats. Report the head/tail trade-off (Section C3), or pre-register a tail-weighted/per-group primary metric before re-evaluating.
4. **Increase replication for close models.** Use at least 5 seeds and report paired confidence intervals or a paired bootstrap. With 2 seeds, `tail_aware` vs `random` and `random` vs `diversity` are not resolved.

### 2. Should fix before final evaluation

5. Add top-k/top-p/repetition penalty to generation, or standardise on the diagnostic’s `--top-k/--top-p` settings for qualitative examples.
6. Decide and document EOT semantics: document separator vs stop token. If a “single paper” mode is intended, stopping at EOT is correct; otherwise do not stop.
7. Decide whether cross-document attention is acceptable. If not, add a document-level attention mask or evaluate the sensitivity of the comparison to packing.
8. Report the `quality` vs `random` overlap (Jaccard 0.948) and `tail_aware` vs `tail_only` overlap (0.946) as a limitation; consider a cleaner ablation with disjoint pools.
9. Promote `eval_by_group.py` from a diagnostic to a maintained evaluation script (Part 4) with per-category and per-theme breakdowns.
10. Add the new generation diagnostics to CI/smoke tests: fixed prompt, `eot2` must reproduce the fusion flag, `none` must not; EOS must be detected by id.

### 3. Optional improvements

11. Train longer or larger for absolute text quality (not for the fixed-budget comparison). At 60M tokens the models are ~5.7× below Chinchilla-optimal for 17M parameters.
12. Add constrained/repetition-aware decoding for demos.
13. Use prompt templates or few-shot examples if the goal is instruction-following; the current models are document LMs.
14. Fix the tokenizer decoder `add_prefix_space` inconsistency (cosmetic; round-trip already passes).
15. Remove dead code and the ineffective trailing-EOT loop in `src/generate.py`; use a repo-relative tokenizer path.

---

## F. Final verdict

**Are the checkpoints loading correctly?**
**Yes.** All 10 `ckpt_best.pt` files load with `strict=True`, zero missing/unexpected keys, identical 17.136M-parameter configs, step 3,662, and validation losses that match the summaries and `reports/part3_report.json`.

**Is the tokenizer behaving correctly?**
**Yes.** 16,384-token byte-level BPE; EOT=0 and PAD=1 distinct; exact round-trip on scientific text, Unicode, LaTeX, leading spaces, double spaces, and newlines; `tests/test_tokenizer.py` passes. The decoder `add_prefix_space=True` vs pre-tokenizer `False` mismatch is cosmetic and did not affect any round-trip. The tokenizer is **not** the cause of the missing spaces or malformed words. The missing space is caused by decoding artificial EOT tokens with `skip_special_tokens=True`.

**Are document boundaries being learned correctly?**
**Yes.** Exactly one EOT per document; EOT is included in targets; 500/500 audited documents have the correct boundary; the models assign `p(EOT) ≈ 0.51–0.57` at true document ends, recall `0.83–0.88`, false-positive rate `≈0.004`. Windows cross document boundaries by design (99.98% contain an internal EOT), which is standard packing but means the model learns to start a new document after EOT — exactly the behavior triggered by the bad prompt suffix.

**Is generation implemented correctly?**
**Partially.** The autoregressive core is correct: `model.eval()`, `torch.no_grad()`, last-position logits, temperature scaling, EOS checked by token id, context cropping. But the **prompt construction** (`[eot, eot]`) and the **decoding** (prompt + artificial EOTs + continuation decoded together with special tokens skipped) are wrong for free-form prompts. This is a genuine bug, not a model-size effect. EOT-as-EOS is also a semantic mismatch. Top-k/top-p/repetition controls are absent.

**Are the models undertrained?**
**Yes, in absolute terms.** 60M tokens for a 17.1M-parameter model is about 3.5 tokens/parameter, roughly 5.7× below Chinchilla-optimal (~343M). Validation loss is still decreasing at the final step, and `best_step = 3662` for every run. They are stable and not overfitting, but they are not converged. Undertraining explains weak/invented words and repetition; it does **not** explain the deterministic `is` + `Title` fusion.

**Are the current generations meaningful enough to use in the report?**
**No, not the `src/generate.py` outputs.** The `eot2` strings are artifacts: 100% of 100 such generations have a fused prompt/title boundary. They must not be quoted as model behavior. The `none` continuations are more meaningful but still show the model’s limitations (invented titles/words, repetition). Use aggregate validation loss and per-group loss for conclusions; use generation only as clearly labelled qualitative examples after the prompt/decoding fix, with multiple sampling seeds.

**Can we trust the existing training comparisons?**
**Mechanically, yes; statistically, only partially.**
- The training comparison is fair: same tokenizer, architecture, hyperparameters, token budget, validation set, and initialization per seed; test split untouched; report matches summaries.
- `quality` is robustly better than `random` in every group and both seeds.
- `tail_only` is robustly worst overall.
- `tail_aware` is **not** better than `random` on overall validation loss; it trades ~0.023 nats of head loss for ~0.104 nats of tail loss. That is a real and potentially desirable trade-off, but it is not a win on the current primary metric.
- With only 2 seeds and 94.8% document overlap between `quality` and `random`, small overall differences should not be over-interpreted. More seeds and per-group evaluation are required before making a final strategy recommendation.

**Bottom line:** the training pipeline and checkpoints are sound; the generation script is not. The strange outputs are explained by two concrete bugs in `src/generate.py` (out-of-distribution `[eot, eot]` prompt suffix and EOT-skipping decode), plus expected limitations of a 17M-parameter, 60M-token document LM. Fix generation before using any generated text, and report the tail/head trade-off instead of relying on overall validation loss alone.
