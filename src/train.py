"""Train one compact GPT on one Part 2 training subset.

Examples
--------
Pilot (200 steps, not part of the comparison):
  python src/train.py --subset random --seed 42 --max-steps 200 \
      --out-dir runs/pilot/random_seed42

Full run:
  python src/train.py --subset tail_aware --seed 42

The test split is never opened here. Use validation loss for monitoring and
keep test.bin for Part 4.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
from data import TokenStream, TokenWindows, steps_for_tokens
from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    subset: str
    seed: int = config.SEED
    max_tokens: int = config.PART3_MAX_TOKENS
    block_size: int = config.PART3_BLOCK_SIZE
    batch_size: int = config.PART3_BATCH_SIZE
    grad_accum: int = config.PART3_GRAD_ACCUM
    lr: float = config.PART3_LR
    min_lr_ratio: float = config.PART3_MIN_LR_RATIO
    warmup_frac: float = config.PART3_WARMUP_FRAC
    weight_decay: float = config.PART3_WEIGHT_DECAY
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = config.PART3_GRAD_CLIP
    dropout: float = config.PART3_DROPOUT
    n_layer: int = config.PART3_N_LAYER
    n_head: int = config.PART3_N_HEAD
    n_embd: int = config.PART3_N_EMBD
    eval_every: int = config.PART3_EVAL_EVERY
    log_every: int = config.PART3_LOG_EVERY
    save_every: int = config.PART3_SAVE_EVERY
    device: str = "auto"
    amp: bool = True
    max_steps: int | None = None
    data_dir: Path = config.SUBSETS_DIR
    out_dir: Path | None = None
    resume: bool = False
    dry_run: bool = False


def _jsonable(value: Any) -> Any:
    """Make dataclass/Path/float values safe for JSON logs."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def resolve_amp(device: torch.device, enabled: bool) -> torch.dtype | None:
    if not enabled or device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is not supported on this GPU; run with --no-amp")
    return torch.bfloat16


def amp_label(dtype: torch.dtype | None) -> str:
    return "none" if dtype is None else str(dtype).replace("torch.", "")


def make_logger(out_dir: Path, resume: bool) -> logging.Logger:
    logger = logging.getLogger(f"part3.{out_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(
        out_dir / "run.log", mode="a" if resume else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"vocab_size": config.VOCAB_SIZE, "eot_id": 0, "pad_id": 1}


def build_optimizer(
    model: GPT, config_: TrainConfig, device: torch.device
) -> AdamW:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config_.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=config_.lr,
        betas=(config_.beta1, config_.beta2),
        eps=1e-8,
        fused=device.type == "cuda",
    )


def build_scheduler(
    optimizer: AdamW, max_steps: int, warmup_frac: float, min_lr_ratio: float
) -> LambdaLR:
    warmup_steps = max(1, int(warmup_frac * max_steps))

    def lr_lambda(step: int) -> float:
        # LambdaLR calls this with step 0 at construction, before any optimizer
        # update. Count completed updates as step + 1 so the first update does
        # not run at learning rate zero.
        current = step + 1
        if current <= warmup_steps:
            return current / warmup_steps
        progress = min(
            1.0, (current - warmup_steps) / max(1, max_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def evaluate(
    model: GPT,
    windows: TokenWindows,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in windows.val_batches(batch_size):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            _, loss = model(x, y)
        assert loss is not None
        total_loss += float(loss.item()) * x.numel()
        total_tokens += x.numel()
    model.train()
    if total_tokens == 0:
        raise RuntimeError("validation set produced no tokens")
    return total_loss / total_tokens


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint on CPU so RNG byte tensors stay CPU tensors.

    ``map_location=device`` would move the saved RNG state to CUDA and make
    ``torch.set_rng_state`` fail on resume. Model and optimizer state are
    copied onto the model's device by their own load_state_dict calls.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def truncate_metrics(path: Path, max_step: int) -> None:
    """Drop metric rows logged after the checkpoint that will be replayed.

    A process killed while appending can leave a partial final line; skip it
    rather than making the resumed run fail to start.
    """
    path = Path(path)
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            step = int(json.loads(line)["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if step <= max_step:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


_RESUME_IGNORED = {"resume", "dry_run", "out_dir", "max_steps"}


def resume_mismatches(saved: dict, current: dict) -> list[str]:
    """Return the training settings that differ from a checkpoint's run."""

    def norm(value: Any) -> Any:
        return str(value) if isinstance(value, Path) else value

    mismatches = []
    for key in sorted(set(saved) | set(current)):
        if key in _RESUME_IGNORED:
            continue
        if norm(saved.get(key)) != norm(current.get(key)):
            mismatches.append(
                f"{key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}"
            )
    return mismatches


def _check_token_range(windows: TokenWindows, vocab_size: int, name: str) -> None:
    maximum = int(windows.data.max())
    if maximum >= vocab_size:
        raise ValueError(
            f"{name} contains token id {maximum}, but vocab_size is {vocab_size}"
        )


def train(config_: TrainConfig, *, model_factory=None, model_metadata=None) -> dict:
    if config_.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if config_.max_steps is not None and config_.max_steps < 1:
        raise ValueError("max_steps must be positive when provided")

    data_dir = Path(config_.data_dir)
    out_dir = (
        Path(config_.out_dir)
        if config_.out_dir is not None
        else config.RUNS_DIR / config_.subset / f"seed{config_.seed}"
    )
    train_path = data_dir / config_.subset / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists():
        raise FileNotFoundError(f"missing training file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"missing validation file: {val_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if not config_.resume:
        # A leftover summary from an interrupted forced rerun would otherwise
        # make train_all and the report treat this run as complete.
        summary_path.unlink(missing_ok=True)
    logger = make_logger(out_dir, config_.resume)
    device = resolve_device(config_.device)
    amp_dtype = resolve_amp(device, config_.amp)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    random.seed(config_.seed)
    np.random.seed(config_.seed)
    torch.manual_seed(config_.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config_.seed)
        torch.cuda.reset_peak_memory_stats()

    meta = load_meta(data_dir)
    vocab_size = int(meta.get("vocab_size", config.VOCAB_SIZE))
    train_stream = TokenStream(
        train_path,
        block_size=config_.block_size,
        batch_size=config_.batch_size,
        grad_accum=config_.grad_accum,
        seed=config_.seed,
    )
    val_windows = TokenWindows(val_path, config_.block_size)
    _check_token_range(train_stream.windows, vocab_size, "train.bin")
    _check_token_range(val_windows, vocab_size, "val.bin")

    max_steps = (
        config_.max_steps
        if config_.max_steps is not None
        else steps_for_tokens(
            config_.max_tokens,
            config_.block_size,
            config_.batch_size,
            config_.grad_accum,
        )
    )
    if max_steps < 1:
        raise ValueError("the token budget is smaller than one optimizer step")
    tokens_per_step = (
        config_.block_size * config_.batch_size * config_.grad_accum
    )
    planned_tokens = max_steps * tokens_per_step
    if model_factory is None:
        model_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=config_.block_size,
            n_layer=config_.n_layer,
            n_head=config_.n_head,
            n_embd=config_.n_embd,
            dropout=config_.dropout,
        )
        model_metadata = asdict(model_config)
        model_factory = lambda: GPT(model_config)
    elif model_metadata is None:
        raise ValueError("custom models require checkpoint metadata")
    run_config = {
        "subset": config_.subset,
        "seed": config_.seed,
        "device": str(device),
        "amp_dtype": amp_label(amp_dtype),
        "model": model_metadata,
        "train": asdict(config_),
        "data": {
            "train_bin": str(train_path),
            "val_bin": str(val_path),
            "train_tokens": len(train_stream.windows.data),
            "val_tokens": len(val_windows.data),
            "vocab_size": vocab_size,
            "eot_id": int(meta.get("eot_id", 0)),
            "pad_id": int(meta.get("pad_id", 1)),
        },
        "max_steps": max_steps,
        "tokens_per_step": tokens_per_step,
        "planned_tokens": planned_tokens,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    config_name = "config_resume.json" if config_.resume else "config.json"
    atomic_write_text(
        json.dumps(_jsonable(run_config), indent=2), out_dir / config_name
    )

    if config_.dry_run:
        logger.info(
            "dry run: subset=%s seed=%d steps=%d tokens=%d",
            config_.subset,
            config_.seed,
            max_steps,
            planned_tokens,
        )
        return {
            "dry_run": True,
            "subset": config_.subset,
            "seed": config_.seed,
            "max_steps": max_steps,
            "tokens_per_step": tokens_per_step,
            "planned_tokens": planned_tokens,
            "out_dir": str(out_dir),
            "train_bin": str(train_path),
            "val_bin": str(val_path),
        }

    logger.info(
        "device=%s amp=%s subset=%s seed=%d",
        device,
        amp_label(amp_dtype),
        config_.subset,
        config_.seed,
    )
    model = model_factory().to(device)
    logger.info(
        "model %.2fM params | steps=%d | tokens/step=%d | planned=%.2fM",
        model.num_parameters() / 1e6,
        max_steps,
        tokens_per_step,
        planned_tokens / 1e6,
    )
    optimizer = build_optimizer(model, config_, device)
    scheduler = build_scheduler(
        optimizer, max_steps, config_.warmup_frac, config_.min_lr_ratio
    )

    start_step = 0
    elapsed_total = 0.0
    previous_peak_mb = 0.0
    best_val = math.inf
    best_step = 0
    resume_path = out_dir / "ckpt_last.pt"
    if config_.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"cannot resume, missing {resume_path}")
        checkpoint = load_checkpoint(resume_path)
        if checkpoint["config"] != model_metadata:
            raise ValueError("checkpoint model config does not match the current run")
        mismatches = resume_mismatches(
            checkpoint.get("train_config", {}), asdict(config_)
        )
        if mismatches:
            raise ValueError(
                "resume settings differ from the checkpoint:\n  "
                + "\n  ".join(mismatches)
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        elapsed_total = float(checkpoint.get("elapsed_s", 0.0))
        previous_peak_mb = float(checkpoint.get("peak_mem_mb", 0.0))
        best_val = float(checkpoint.get("best_val", math.inf))
        best_step = int(checkpoint.get("best_step", 0))
        train_stream.skip_steps(start_step)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        logger.info("resumed from %s at step %d", resume_path, start_step)

    if start_step > max_steps:
        raise ValueError(
            f"checkpoint step {start_step} exceeds the current max_steps {max_steps}"
        )

    metrics_path = out_dir / "metrics.jsonl"
    if config_.resume:
        truncate_metrics(metrics_path, start_step)
    metrics_mode = "a" if config_.resume else "w"
    metrics_file = open(metrics_path, metrics_mode, encoding="utf-8")

    def peak_memory_mb() -> float:
        if device.type != "cuda":
            return previous_peak_mb
        return max(previous_peak_mb, torch.cuda.max_memory_allocated() / 1024**2)

    def write_metric(record: dict) -> None:
        metrics_file.write(json.dumps(_jsonable(record)) + "\n")
        metrics_file.flush()

    def save_last(step: int, tokens: int, val_loss: float | None) -> None:
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "tokens_seen": tokens,
                "best_val": best_val,
                "best_step": best_step,
                "config": model_metadata,
                "train_config": asdict(config_),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "val_loss": _finite_or_none(val_loss),
                "elapsed_s": elapsed_total + (time.perf_counter() - wall_start),
                "peak_mem_mb": peak_memory_mb(),
            },
            out_dir / "ckpt_last.pt",
        )

    def save_best(step: int, tokens: int, val_loss: float) -> None:
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "config": model_metadata,
                "step": step,
                "tokens_seen": tokens,
                "val_loss": val_loss,
            },
            out_dir / "ckpt_best.pt",
        )

    logger.info("max_steps=%d planned_tokens=%d", max_steps, planned_tokens)
    logger.info("validation windows=%d", val_windows.n_windows)

    # Evaluate the initial model so the learning curve starts at step 0. A
    # resumed run keeps the rows already in metrics.jsonl instead of repeating
    # this evaluation.
    if start_step == 0:
        val_loss = evaluate(model, val_windows, config_.batch_size, device, amp_dtype)
        if val_loss < best_val:
            best_val = val_loss
            best_step = start_step
            save_best(start_step, start_step * tokens_per_step, val_loss)
        write_metric(
            {
                "step": start_step,
                "tokens": start_step * tokens_per_step,
                "train_loss": None,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": None,
                "elapsed_s": 0.0,
                "tokens_per_s": None,
                "peak_mem_mb": peak_memory_mb(),
                "val_loss": val_loss,
                "val_ppl": _finite_or_none(math.exp(val_loss)),
                "best_val": _finite_or_none(best_val),
            }
        )
        logger.info("step 0 val_loss=%.4f val_ppl=%.2f", val_loss, math.exp(val_loss))
    else:
        logger.info(
            "resumed at step %d; step-0 validation not repeated (best_val=%.4f)",
            start_step,
            best_val,
        )

    wall_start = time.perf_counter()
    val_loss: float | None = None
    for step in range(start_step, max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config_.grad_accum):
            x, y = train_stream.next_micro_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast(device, amp_dtype):
                _, loss = model(x, y)
            assert loss is not None
            scaled = loss / config_.grad_accum
            scaled.backward()
            loss_sum += float(scaled.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config_.grad_clip)
        optimizer.step()
        scheduler.step()

        completed = step + 1
        tokens_seen = completed * tokens_per_step
        lr = scheduler.get_last_lr()[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = elapsed_total + (time.perf_counter() - wall_start)
        peak_mem = peak_memory_mb()

        log_due = completed % config_.log_every == 0 or completed == max_steps
        eval_due = completed % config_.eval_every == 0 or completed == max_steps
        val_loss = None
        if eval_due:
            val_loss = evaluate(
                model, val_windows, config_.batch_size, device, amp_dtype
            )
            if val_loss < best_val:
                best_val = val_loss
                best_step = completed
                save_best(completed, tokens_seen, val_loss)
            logger.info(
                "step %d/%d train_loss=%.4f val_loss=%.4f val_ppl=%.2f lr=%.2e",
                completed,
                max_steps,
                loss_sum,
                val_loss,
                math.exp(val_loss),
                lr,
            )

        if log_due or eval_due:
            write_metric(
                {
                    "step": completed,
                    "tokens": tokens_seen,
                    "train_loss": loss_sum,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "elapsed_s": elapsed,
                    "tokens_per_s": tokens_seen / elapsed if elapsed > 0 else None,
                    "peak_mem_mb": peak_mem,
                    "val_loss": val_loss,
                    "val_ppl": (
                        _finite_or_none(math.exp(val_loss))
                        if val_loss is not None
                        else None
                    ),
                    "best_val": _finite_or_none(best_val),
                }
            )
            if log_due and not eval_due:
                logger.info(
                    "step %d/%d train_loss=%.4f lr=%.2e grad_norm=%.3f tokens/s=%.0f",
                    completed,
                    max_steps,
                    loss_sum,
                    lr,
                    float(grad_norm),
                    tokens_seen / elapsed if elapsed > 0 else 0.0,
                )

        if completed % config_.save_every == 0 or completed == max_steps:
            save_last(completed, tokens_seen, val_loss)

    # The last training step always evaluates, so reuse that value instead of
    # running a second full validation pass over the same weights.
    if val_loss is None:
        final_val = evaluate(model, val_windows, config_.batch_size, device, amp_dtype)
    else:
        final_val = val_loss
    if final_val < best_val:
        best_val = final_val
        best_step = max_steps
        save_best(max_steps, max_steps * tokens_per_step, final_val)
    final_tokens = max_steps * tokens_per_step
    wall_time = elapsed_total + (time.perf_counter() - wall_start)
    peak_mem = peak_memory_mb()

    atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": model_metadata,
            "step": max_steps,
            "tokens_seen": final_tokens,
            "val_loss": final_val,
        },
        out_dir / "model_last.pt",
    )
    save_last(max_steps, final_tokens, final_val)

    summary = {
        "subset": config_.subset,
        "seed": config_.seed,
        "step": max_steps,
        "tokens_seen": final_tokens,
        "planned_tokens": planned_tokens,
        "tokens_per_step": tokens_per_step,
        "model_params": model.num_parameters(),
        "final_val_loss": final_val,
        "final_val_ppl": _finite_or_none(math.exp(final_val)),
        "best_val_loss": _finite_or_none(best_val),
        "best_step": best_step,
        "wall_time_s": wall_time,
        "tokens_per_s": final_tokens / wall_time if wall_time > 0 else None,
        "peak_mem_mb": peak_mem,
        "device": str(device),
        "amp_dtype": amp_label(amp_dtype),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }
    atomic_write_text(json.dumps(_jsonable(summary), indent=2), summary_path)
    metrics_file.close()
    logger.info(
        "finished: final_val_loss=%.4f final_val_ppl=%.2f best_val_loss=%.4f "
        "wall=%.1fs tokens/s=%.0f",
        final_val,
        math.exp(final_val),
        best_val,
        wall_time,
        final_tokens / wall_time if wall_time > 0 else 0.0,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, help="Part 2 subset name")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-tokens", type=int, default=config.PART3_MAX_TOKENS)
    parser.add_argument("--block-size", type=int, default=config.PART3_BLOCK_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.PART3_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=config.PART3_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=config.PART3_LR)
    parser.add_argument("--min-lr-ratio", type=float, default=config.PART3_MIN_LR_RATIO)
    parser.add_argument("--warmup-frac", type=float, default=config.PART3_WARMUP_FRAC)
    parser.add_argument("--weight-decay", type=float, default=config.PART3_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=config.PART3_GRAD_CLIP)
    parser.add_argument("--dropout", type=float, default=config.PART3_DROPOUT)
    parser.add_argument("--n-layer", type=int, default=config.PART3_N_LAYER)
    parser.add_argument("--n-head", type=int, default=config.PART3_N_HEAD)
    parser.add_argument("--n-embd", type=int, default=config.PART3_N_EMBD)
    parser.add_argument("--eval-every", type=int, default=config.PART3_EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=config.PART3_LOG_EVERY)
    parser.add_argument("--save-every", type=int, default=config.PART3_SAVE_EVERY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=config.SUBSETS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configuration = TrainConfig(
        subset=args.subset,
        seed=args.seed,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_frac=args.warmup_frac,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        eval_every=args.eval_every,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
        amp=not args.no_amp,
        max_steps=args.max_steps,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    summary = train(configuration)
    if configuration.dry_run:
        print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
