"""Fair, token-level diagnostic for the five Part 3 checkpoints.

This script is read-only: it never writes checkpoints and never touches the
corpus, tokenizer, or training code. It exists because the earlier ad-hoc
analysis gave a different prompt to every model, which made the outputs
incomparable.

What it does
------------
* Loads every selected checkpoint with ``GPTConfig(**ckpt["config"])`` and
  ``load_state_dict(..., strict=True)``.
* Uses one fixed prompt set for every model.
* Uses identical decoding settings for every model/prompt pair.
* Records prompt and continuation *separately*, plus the raw token ids.
* Detects EOS by token id (never by looking for the string
  ``"<|endoftext|>"`` in decoded text).
* Reports whether generation stopped at EOS or at the token limit.
* Measures simple, explicitly-labelled repetition statistics.
* Computes a fixed validation-loss comparison on the same windows for every
  checkpoint (so the loss comparison is not affected by generation settings).
* Writes JSON + CSV and prints a comparison table.

Prompt construction is configurable because the current ``src/generate.py``
appends two EOT tokens after the prompt. ``--prompt-suffix eot2`` reproduces
that behaviour (default, for comparability with existing outputs); ``none``
conditions on the prompt only, which is the clean continuation baseline.

Example
-------
    uv run --group train python deep_analysis.py
    uv run --group train python deep_analysis.py --prompt-suffix none
    uv run --group train python deep_analysis.py --decoding greedy
    uv run --group train python deep_analysis.py --seeds 42 43 --sample-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import TokenWindows  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

MODELS = ("random", "quality", "diversity", "tail_aware", "tail_only")

# One fixed prompt set for every model. These are deliberately generic
# scientific continuations, plus two title-like prompts, so the comparison
# exercises both sentence continuation and document-start behaviour.
PROMPTS = (
    "The key innovation in machine learning is",
    "Recent advances in large language models",
    "Transformers have demonstrated",
    "We propose a method for",
    "Rare events in scientific texts",
)

PROMPT_SUFFIXES = ("eot2", "eot1", "none", "newline")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def build_prefix(tok: Tokenizer, prompt: str, suffix: str, block_size: int):
    """Return (prompt_ids, prefix_ids, suffix_ids, truncated).

    ``prefix_ids`` is exactly what the model conditions on. The ``eot2`` suffix
    deliberately reproduces the **pre-fix** ``src/generate.py`` construction
    (``prompt_ids + [eot, eot]``); the current production generator never builds
    this prefix and uses :func:`src.generate.build_generation_prefix` instead.
    """
    eot = tok.token_to_id("<|endoftext|>")
    prompt_ids = list(tok.encode(prompt).ids)
    if suffix == "eot2":
        suffix_ids = [eot, eot]
    elif suffix == "eot1":
        suffix_ids = [eot]
    elif suffix == "none":
        suffix_ids = []
    elif suffix == "newline":
        suffix_ids = list(tok.encode("\n\n").ids)
    else:
        raise ValueError(f"unknown prompt suffix: {suffix}")

    room = block_size - len(suffix_ids)
    if room < 1:
        raise ValueError("prompt suffix leaves no room for the prompt")
    truncated = len(prompt_ids) > room
    prompt_ids = prompt_ids[:room]
    return prompt_ids, prompt_ids + suffix_ids, suffix_ids, truncated


def _filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -math.inf), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -math.inf)
        logits = torch.full_like(logits, -math.inf).scatter(0, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_ids(
    model: GPT,
    prefix_ids: list[int],
    *,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_k: int,
    top_p: float,
    eot_id: int,
    sample_seed: int,
    device: torch.device,
) -> dict:
    """Autoregressive generation that records every sampled token id.

    EOS is handled by id. The generated EOS id is *recorded* but not appended
    to the context and not decoded as text.
    """
    torch.manual_seed(sample_seed)
    block_size = model.config.block_size
    ids = list(prefix_ids)
    generated: list[int] = []
    sampled_probs: list[float] = []
    stop_reason = "max_new_tokens"

    for _ in range(max_new_tokens):
        if len(ids) > block_size:
            # Match src/generate.py: slide the context window.
            ids = ids[-block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(idx)
        next_logits = logits[0, -1, :].float()

        if decoding == "greedy":
            next_id = int(torch.argmax(next_logits).item())
            prob = float(torch.softmax(next_logits, dim=-1)[next_id].item())
        elif decoding == "sample":
            if temperature <= 0:
                raise ValueError("temperature must be > 0 for sampling")
            next_logits = next_logits / temperature
            next_logits = _filter_top_k_top_p(next_logits, top_k, top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            prob = float(probs[next_id].item())
        else:
            raise ValueError(f"unknown decoding mode: {decoding}")

        generated.append(next_id)
        sampled_probs.append(prob)
        if next_id == eot_id:
            stop_reason = "eos"
            break
        ids.append(next_id)

    continuation = [t for t in generated if t != eot_id]
    return {
        "generated_token_ids": generated,
        "continuation_token_ids": continuation,
        "eos_position": (
            generated.index(eot_id) if eot_id in generated else None
        ),
        "stop_reason": stop_reason,
        "sampled_probs": sampled_probs,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ngram_stats(seq: list, n: int) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    if len(seq) < n:
        return {
            "n": n,
            "total": 0,
            "distinct": 0,
            "distinct_ratio": None,
            "repeat_fraction": None,
            "max_count": 0,
        }
    grams = Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    total = len(seq) - n + 1
    distinct = len(grams)
    return {
        "n": n,
        "total": total,
        "distinct": distinct,
        "distinct_ratio": distinct / total,
        "repeat_fraction": 1.0 - distinct / total,
        "max_count": max(grams.values()),
    }


def longest_same_token_run(seq: list[int]) -> int:
    best = run = 0
    previous = None
    for token in seq:
        if token == previous:
            run += 1
        else:
            run = 1
            previous = token
        best = max(best, run)
    return best


def repetition_metrics(continuation_ids: list[int], text: str) -> dict:
    words = text.split()
    return {
        "tokens": len(continuation_ids),
        "chars": len(text),
        "words": len(words),
        "type_token_ratio_words": (len(set(words)) / len(words)) if words else None,
        "token_2gram": ngram_stats(continuation_ids, 2),
        "token_3gram": ngram_stats(continuation_ids, 3),
        "token_4gram": ngram_stats(continuation_ids, 4),
        "word_3gram": ngram_stats(words, 3),
        "word_4gram": ngram_stats(words, 4),
        "longest_same_token_run": longest_same_token_run(continuation_ids),
    }


def boundary_joined(prompt_text: str, continuation_text: str) -> bool:
    """True when decoding fused the prompt and continuation with no separator."""
    if not prompt_text or not continuation_text:
        return False
    return prompt_text[-1].isalnum() and continuation_text[0].isalnum()


# ---------------------------------------------------------------------------
# Validation loss on a fixed window set
# ---------------------------------------------------------------------------
@torch.no_grad()
def fixed_val_loss(
    model: GPT,
    path: Path,
    *,
    n_windows: int,
    batch_size: int,
    device: torch.device,
) -> float:
    windows = TokenWindows(path, model.config.block_size)
    limit = min(n_windows, windows.n_windows)
    # Evenly spaced windows across the whole val file, so the estimate is not
    # dominated by the first few documents (val.bin is sorted by arXiv id).
    indices = np.unique(
        np.linspace(0, windows.n_windows - 1, limit).round().astype(np.int64)
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp = torch.bfloat16 if device.type == "cuda" else None
    for start in range(0, len(indices), batch_size):
        x, y = windows.get_batch(indices[start : start + batch_size])
        x = x.to(device)
        y = y.to(device)
        if amp is not None:
            with torch.autocast(device_type=device.type, dtype=amp):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        assert loss is not None
        total_loss += float(loss.item()) * x.numel()
        total_tokens += x.numel()
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    parser.add_argument("--prompt-suffix", nargs="+", choices=PROMPT_SUFFIXES, default=["eot2", "none"])
    parser.add_argument("--decoding", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 disables top-p")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--sample-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--val-windows", type=int, default=512,
                        help="number of evenly spaced validation windows for the loss check (0 = skip)")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "deep_analysis")
    args = parser.parse_args(argv)

    if args.temperature <= 0 and args.decoding == "sample":
        parser.error("--temperature must be > 0 for sampling")
    if not (0.0 < args.top_p <= 1.0):
        parser.error("--top-p must be in (0, 1]")

    device = resolve_device(args.device)
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    eot_id = tok.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise RuntimeError("tokenizer has no <|endoftext|> token")
    val_path = ROOT / "data" / "subsets" / "val.bin"

    tokenizer_meta = json.loads((ROOT / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_record = {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id,
        "pad_id": tok.token_to_id("<|pad|>"),
        "decode_eot_with_skip": tok.decode([eot_id]),
        "decode_eot_keep": tok.decode([eot_id], skip_special_tokens=False),
        "normalizer": tokenizer_meta.get("normalizer"),
        "pre_tokenizer": tokenizer_meta.get("pre_tokenizer"),
        "decoder": tokenizer_meta.get("decoder"),
    }

    records: list[dict] = []
    checkpoint_records: list[dict] = []
    val_losses: dict[tuple[str, int], float | None] = {}

    for subset in args.models:
        for seed in args.seeds:
            ckpt_path = ROOT / "runs" / subset / f"seed{seed}" / "ckpt_best.pt"
            if not ckpt_path.exists():
                print(f"WARNING missing checkpoint: {ckpt_path}", file=sys.stderr)
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = GPTConfig(**ckpt["config"])
            model = GPT(cfg)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device)
            model.eval()

            val_loss = None
            if args.val_windows > 0 and val_path.exists():
                val_loss = fixed_val_loss(
                    model, val_path,
                    n_windows=args.val_windows,
                    batch_size=args.val_batch_size,
                    device=device,
                )
            val_losses[(subset, seed)] = val_loss

            checkpoint_records.append({
                "model": subset,
                "seed": seed,
                "checkpoint": str(ckpt_path),
                "step": ckpt.get("step"),
                "tokens_seen": ckpt.get("tokens_seen"),
                "checkpoint_val_loss": ckpt.get("val_loss"),
                "fixed_val_loss": val_loss,
                "params": model.num_parameters(),
                "config": cfg.__dict__,
                "state_dict_missing_keys": [],
                "state_dict_unexpected_keys": [],
            })
            fixed_text = f", fixed_val={val_loss:.6f}" if val_loss is not None else ""
            print(
                f"loaded {subset}/seed{seed}: {model.num_parameters()/1e6:.3f}M params, "
                f"step={ckpt.get('step')}, ckpt_val={ckpt.get('val_loss')}{fixed_text}",
                flush=True,
            )

            for prompt in args.prompts:
                for suffix in args.prompt_suffix:
                    for sample_seed in args.sample_seeds:
                        prompt_ids, prefix_ids, suffix_ids, truncated = build_prefix(
                            tok, prompt, suffix, cfg.block_size
                        )
                        result = generate_ids(
                            model, prefix_ids,
                            max_new_tokens=args.max_new_tokens,
                            decoding=args.decoding,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            eot_id=eot_id,
                            sample_seed=sample_seed,
                            device=device,
                        )

                        continuation_text = tok.decode(
                            result["continuation_token_ids"], skip_special_tokens=True
                        )
                        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
                        full_skip = tok.decode(
                            prefix_ids + result["generated_token_ids"],
                            skip_special_tokens=True,
                        )
                        full_keep = tok.decode(
                            prefix_ids + result["generated_token_ids"],
                            skip_special_tokens=False,
                        )
                        continuation_without_leading_space = continuation_text.lstrip()
                        piece_list = [tok.decode([t]) for t in result["generated_token_ids"][:32]]

                        record = {
                            "model": subset,
                            "seed": seed,
                            "prompt_index": args.prompts.index(prompt),
                            "prompt": prompt,
                            "prompt_suffix": suffix,
                            "decoding": args.decoding,
                            "temperature": args.temperature,
                            "top_k": args.top_k,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "sample_seed": sample_seed,
                            "prompt_token_ids": prompt_ids,
                            "prompt_suffix_token_ids": suffix_ids,
                            "prefix_token_ids": prefix_ids,
                            "prompt_truncated": truncated,
                            "generated_token_ids": result["generated_token_ids"],
                            "continuation_token_ids": result["continuation_token_ids"],
                            "eos_id": eot_id,
                            "eos_position": result["eos_position"],
                            "eos_generated": result["eos_position"] is not None,
                            "stop_reason": result["stop_reason"],
                            "generated_token_count_including_eos": len(result["generated_token_ids"]),
                            "continuation_token_count": len(result["continuation_token_ids"]),
                            "prompt_text": prompt_text,
                            "continuation_text": continuation_text,
                            "full_text_skip_special": full_skip,
                            "full_text_keep_special": full_keep,
                            "prompt_char_len": len(prompt_text),
                            "continuation_char_len": len(continuation_text),
                            "boundary_joined": boundary_joined(prompt_text, continuation_text),
                            "prompt_plus_continuation_equals_full_skip": (
                                full_skip == prompt_text + continuation_text
                            ),
                            "first_32_generated_pieces": piece_list,
                            "mean_sampled_prob": (
                                float(np.mean(result["sampled_probs"]))
                                if result["sampled_probs"] else None
                            ),
                            "min_sampled_prob": (
                                float(np.min(result["sampled_probs"]))
                                if result["sampled_probs"] else None
                            ),
                            "repetition": repetition_metrics(
                                result["continuation_token_ids"], continuation_text
                            ),
                        }
                        records.append(record)
                        print(
                            f"  {subset}/s{seed} {suffix:7} prompt[{args.prompts.index(prompt)}] "
                            f"seed={sample_seed} tokens={len(result['continuation_token_ids']):3d} "
                            f"stop={result['stop_reason']:14} boundary_joined={record['boundary_joined']} "
                            f"3gram_repeat={record['repetition']['token_3gram']['repeat_fraction']}",
                            flush=True,
                        )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Aggregate per model x prompt construction.
    summary = []
    for subset in args.models:
        for suffix in args.prompt_suffix:
            rows = [r for r in records if r["model"] == subset and r["prompt_suffix"] == suffix]
            if not rows:
                continue
            def mean_of(path, default=None):
                values = []
                for r in rows:
                    cur = r
                    for key in path:
                        cur = cur.get(key) if isinstance(cur, dict) else None
                    if cur is not None:
                        values.append(float(cur))
                return float(np.mean(values)) if values else default

            summary.append({
                "model": subset,
                "prompt_suffix": suffix,
                "generations": len(rows),
                "eos_rate": float(np.mean([r["eos_generated"] for r in rows])),
                "boundary_joined_rate": float(np.mean([r["boundary_joined"] for r in rows])),
                "mean_continuation_tokens": float(np.mean([r["continuation_token_count"] for r in rows])),
                "mean_continuation_chars": float(np.mean([r["continuation_char_len"] for r in rows])),
                "mean_token_3gram_repeat_fraction": mean_of(("repetition", "token_3gram", "repeat_fraction")),
                "mean_token_4gram_repeat_fraction": mean_of(("repetition", "token_4gram", "repeat_fraction")),
                "max_token_4gram_count": int(max(r["repetition"]["token_4gram"]["max_count"] for r in rows)),
                "mean_word_type_token_ratio": mean_of(("repetition", "type_token_ratio_words")),
                "mean_longest_same_token_run": mean_of(("repetition", "longest_same_token_run")),
                "checkpoint_val_loss_mean": (
                    float(np.mean(ckpt_vals)) if (ckpt_vals := [
                        c["checkpoint_val_loss"] for c in checkpoint_records
                        if c["model"] == subset and c["checkpoint_val_loss"] is not None
                    ]) else None
                ),
                "fixed_val_loss_mean": (
                    float(np.mean(fixed_vals))
                    if (fixed_vals := [
                        val_losses[(subset, seed)]
                        for seed in args.seeds
                        if val_losses.get((subset, seed)) is not None
                    ]) else None
                ),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "models": args.models,
            "seeds": args.seeds,
            "prompts": args.prompts,
            "prompt_suffixes": args.prompt_suffix,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "sample_seeds": args.sample_seeds,
            "val_windows": args.val_windows,
            "device": str(device),
            "note": "top-k/top-p are implemented in this diagnostic only; "
                    "src/generate.py currently exposes temperature only.",
        },
        "tokenizer": tokenizer_record,
        "checkpoints": checkpoint_records,
        "summary": summary,
        "records": records,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat_fields = [
        "model", "seed", "prompt_index", "prompt", "prompt_suffix", "decoding",
        "temperature", "top_k", "top_p", "max_new_tokens", "sample_seed",
        "prompt_truncated", "generated_token_ids", "continuation_token_ids",
        "eos_id", "eos_position", "eos_generated", "stop_reason",
        "continuation_token_count", "prompt_char_len", "continuation_char_len",
        "boundary_joined", "prompt_plus_continuation_equals_full_skip",
        "mean_sampled_prob", "prompt_text", "continuation_text",
        "full_text_skip_special",
        "token_3gram_repeat_fraction", "token_4gram_repeat_fraction",
        "token_4gram_max_count", "word_4gram_repeat_fraction",
        "longest_same_token_run", "type_token_ratio_words",
    ]
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "model": r["model"],
                "seed": r["seed"],
                "prompt_index": r["prompt_index"],
                "prompt": r["prompt"],
                "prompt_suffix": r["prompt_suffix"],
                "decoding": r["decoding"],
                "temperature": r["temperature"],
                "top_k": r["top_k"],
                "top_p": r["top_p"],
                "max_new_tokens": r["max_new_tokens"],
                "sample_seed": r["sample_seed"],
                "prompt_truncated": r["prompt_truncated"],
                "generated_token_ids": " ".join(map(str, r["generated_token_ids"])),
                "continuation_token_ids": " ".join(map(str, r["continuation_token_ids"])),
                "eos_id": r["eos_id"],
                "eos_position": r["eos_position"],
                "eos_generated": r["eos_generated"],
                "stop_reason": r["stop_reason"],
                "continuation_token_count": r["continuation_token_count"],
                "prompt_char_len": r["prompt_char_len"],
                "continuation_char_len": r["continuation_char_len"],
                "boundary_joined": r["boundary_joined"],
                "prompt_plus_continuation_equals_full_skip": r["prompt_plus_continuation_equals_full_skip"],
                "mean_sampled_prob": r["mean_sampled_prob"],
                "prompt_text": r["prompt_text"],
                "continuation_text": r["continuation_text"],
                "full_text_skip_special": r["full_text_skip_special"],
                "token_3gram_repeat_fraction": r["repetition"]["token_3gram"]["repeat_fraction"],
                "token_4gram_repeat_fraction": r["repetition"]["token_4gram"]["repeat_fraction"],
                "token_4gram_max_count": r["repetition"]["token_4gram"]["max_count"],
                "word_4gram_repeat_fraction": r["repetition"]["word_4gram"]["repeat_fraction"],
                "longest_same_token_run": r["repetition"]["longest_same_token_run"],
                "type_token_ratio_words": r["repetition"]["type_token_ratio_words"],
            })

    summary_fields = list(summary[0].keys()) if summary else []
    with out.with_name(out.name + "_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary_fields:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary)

    print("\n" + "=" * 100)
    print("SUMMARY (same prompts and settings for every model)")
    print("=" * 100)
    header = (
        f"{'model':11} {'suffix':7} {'n':>3} {'eos%':>6} {'joined%':>8} "
        f"{'tok':>6} {'tok3rep':>8} {'tok4rep':>8} {'4gmax':>6} "
        f"{'ckptVal':>9} {'fixVal':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['model']:11} {row['prompt_suffix']:7} {row['generations']:3d} "
            f"{100*row['eos_rate']:6.1f} {100*row['boundary_joined_rate']:8.1f} "
            f"{row['mean_continuation_tokens']:6.1f} "
            f"{row['mean_token_3gram_repeat_fraction'] or 0:8.3f} "
            f"{row['mean_token_4gram_repeat_fraction'] or 0:8.3f} "
            f"{row['max_token_4gram_count']:6d} "
            f"{row['checkpoint_val_loss_mean'] if row['checkpoint_val_loss_mean'] is not None else float('nan'):9.4f} "
            f"{row['fixed_val_loss_mean'] if row['fixed_val_loss_mean'] is not None else float('nan'):9.4f}"
        )
    print(f"\nwrote {out.with_suffix('.json')}")
    print(f"wrote {out.with_suffix('.csv')}")
    print(f"wrote {out.with_name(out.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
