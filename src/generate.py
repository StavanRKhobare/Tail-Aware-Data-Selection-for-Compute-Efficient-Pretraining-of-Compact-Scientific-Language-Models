"""Generate text from a trained Part 3 checkpoint.

Two explicit modes are supported:

``continue``
    Condition on the user prompt exactly as encoded. No end-of-text token is
    appended. Generation stops if the model emits EOT.

``document``
    Start from exactly one EOT token (the boundary after a completed document)
    and generate the next document. Generation stops at the next EOT.

Prompts and continuations are decoded separately. Artificial EOT ids are never
decoded into the returned text, so skipping special tokens cannot fuse a prompt
with a generated title.

Examples
--------
python src/generate.py --checkpoint runs/tail_aware/seed42/ckpt_best.pt \
    --prompt "Once upon a time" --max-tokens 150

python src/generate.py --checkpoint runs/quality/seed42/ckpt_best.pt \
    --mode document --max-tokens 150 --seed 42

python -c "
from generate import generate_from_checkpoint
generate_from_checkpoint('runs/quality/seed42/ckpt_best.pt',
                         'Quantum mechanics', max_new_tokens=200)
"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizers import Tokenizer

import config


@dataclass(frozen=True)
class GenerationResult:
    """Separately decoded prompt and continuation plus token-level metadata."""

    mode: str
    prompt_text: str
    continuation_text: str
    text: str
    prompt_token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    suffix_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    eot_id: int
    eos_position: int | None
    stop_reason: str
    prompt_truncated: bool
    seed: int | None
    temperature: float
    decoding: str

    @property
    def eos_generated(self) -> bool:
        return self.eos_position is not None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["eos_generated"] = self.eos_generated
        return payload


def load_checkpoint(path: Path) -> dict:
    """Load a :obj:`ckpt_best.pt` or :obj:`model_last.pt` file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError("checkpoint missing 'config' key")
    return ckpt


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def resolve_tokenizer_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return config.ROOT / "tokenizer" / "tokenizer.json"


def build_generation_prefix(
    tok: Tokenizer,
    prompt: str,
    *,
    mode: str,
    block_size: int,
) -> tuple[list[int], list[int], list[int], bool]:
    """Return ``(prompt_ids, prefix_ids, suffix_ids, prompt_truncated)``.

    ``continue`` uses exactly the encoded prompt and reserves one context slot
    for generation. Long prompts keep their most recent tokens.

    ``document`` starts from exactly one EOT id. A prompt is not allowed in this
    mode, because the mode is specifically the boundary-to-next-document case.

    ``suffix_ids`` is always empty: ``prefix_ids`` is already the complete
    conditioning sequence, so concatenating the two can never duplicate an EOT.
    """
    if mode not in ("continue", "document"):
        raise ValueError(f"unknown generation mode: {mode!r}")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise RuntimeError("tokenizer has no <|endoftext|> token")

    if mode == "document":
        if prompt != "":
            raise ValueError("document mode requires an empty prompt")
        return [], [eot], [], False

    prompt_ids = list(tok.encode(prompt).ids)
    if not prompt_ids:
        raise ValueError(
            "continue mode requires a non-empty prompt; use mode='document' "
            "to generate an unconditional document"
        )

    room = block_size - 1  # reserve one slot for the first generated token
    truncated = len(prompt_ids) > room
    if truncated:
        prompt_ids = prompt_ids[-room:]
    return prompt_ids, list(prompt_ids), [], truncated


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
def generate_with_model(
    model,
    tok: Tokenizer,
    prompt: str,
    *,
    mode: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    decoding: str = "sample",
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = 42,
) -> GenerationResult:
    """Generate from an already loaded model and tokenizer."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if decoding not in ("sample", "greedy"):
        raise ValueError(f"unknown decoding mode: {decoding!r}")
    if decoding == "sample" and temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise RuntimeError("tokenizer has no <|endoftext|> token")

    block_size = int(model.config.block_size)
    prompt_ids, prefix_ids, suffix_ids, truncated = build_generation_prefix(
        tok, prompt, mode=mode, block_size=block_size
    )

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        ids = list(prefix_ids)
        generated: list[int] = []
        stop_reason = "max_new_tokens"

        for _ in range(max_new_tokens):
            if len(ids) > block_size:
                ids = ids[-block_size:]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(idx)
            next_logits = logits[0, -1, :].float()

            if decoding == "greedy":
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                next_logits = _filter_top_k_top_p(next_logits, top_k, top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(
                    torch.multinomial(probs, num_samples=1, generator=generator).item()
                )

            generated.append(next_id)
            if next_id == eot:
                stop_reason = "eos"
                break
            ids.append(next_id)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    continuation_ids = [token for token in generated if token != eot]
    prompt_text = tok.decode(prompt_ids, skip_special_tokens=False) if prompt_ids else ""
    continuation_text = (
        tok.decode(continuation_ids, skip_special_tokens=False)
        if continuation_ids
        else ""
    )
    text = prompt_text + continuation_text if mode == "continue" else continuation_text
    eos_position = generated.index(eot) if eot in generated else None

    return GenerationResult(
        mode=mode,
        prompt_text=prompt_text,
        continuation_text=continuation_text,
        text=text,
        prompt_token_ids=tuple(prompt_ids),
        prefix_token_ids=tuple(prefix_ids),
        suffix_token_ids=tuple(suffix_ids),
        generated_token_ids=tuple(generated),
        continuation_token_ids=tuple(continuation_ids),
        eot_id=eot,
        eos_position=eos_position,
        stop_reason=stop_reason,
        prompt_truncated=truncated,
        seed=seed,
        temperature=temperature,
        decoding=decoding,
    )


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    device: str = "auto",
    mode: str = "continue",
    decoding: str = "sample",
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = 42,
    return_result: bool = False,
    tokenizer_path: str | Path | None = None,
) -> str | GenerationResult:
    """Generate text from a saved model checkpoint.

    By default returns ``result.text``, preserving the original string API.
    Pass ``return_result=True`` to receive the full :class:`GenerationResult`.
    """
    ckpt = load_checkpoint(Path(checkpoint_path))
    cfg = GPTConfig(**ckpt["config"])
    model_device = resolve_device(device)
    model = GPT(cfg).to(model_device)
    model.load_state_dict(ckpt["model"], strict=True)

    tok = Tokenizer.from_file(str(resolve_tokenizer_path(tokenizer_path)))
    result = generate_with_model(
        model,
        tok,
        prompt,
        mode=mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        decoding=decoding,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
    return result if return_result else result.text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to ckpt_best.pt or model_last.pt (e.g. runs/tail_aware/seed42/ckpt_best.pt)",
    )
    parser.add_argument("--prompt", default="", help="Prompt for continue mode.")
    parser.add_argument(
        "--mode",
        choices=("continue", "document"),
        default=None,
        help="continue: prompt-only continuation; document: one EOT then a new document. "
        "Default: document for an empty prompt, continue otherwise.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (>0).")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 disables top-p.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--decoding", choices=("sample", "greedy"), default="sample", help="Decoding strategy."
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device to run on."
    )
    parser.add_argument("--json", action="store_true", help="Print token-level metadata as JSON.")
    args = parser.parse_args(argv)

    mode = args.mode or ("document" if args.prompt == "" else "continue")
    if mode == "document" and args.prompt != "":
        parser.error("--mode document requires an empty --prompt")
    if mode == "continue" and args.prompt == "":
        parser.error("--mode continue requires a non-empty --prompt")

    result = generate_from_checkpoint(
        args.checkpoint,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        mode=mode,
        decoding=args.decoding,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        return_result=True,
    )
    assert isinstance(result, GenerationResult)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.text)


if __name__ == "__main__":
    main()
