"""Run the complete Part 3 training schedule, sequentially.

Each subset/seed pair is an independent run. Completed runs are skipped,
interrupted runs resume from ckpt_last.pt, and the first failure stops the
schedule so a broken run is not silently followed by nine more.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config

SUBSETS = ("random", "quality", "diversity", "tail_aware", "tail_only")

# Resolve every hyperparameter once, at schedule launch, and pass it to each
# child. A mid-schedule edit to config.py or train.py defaults then cannot
# silently mix different settings across the ten runs.
TRAIN_SETTINGS = {
    "--block-size": config.PART3_BLOCK_SIZE,
    "--batch-size": config.PART3_BATCH_SIZE,
    "--grad-accum": config.PART3_GRAD_ACCUM,
    "--lr": config.PART3_LR,
    "--min-lr-ratio": config.PART3_MIN_LR_RATIO,
    "--warmup-frac": config.PART3_WARMUP_FRAC,
    "--weight-decay": config.PART3_WEIGHT_DECAY,
    "--grad-clip": config.PART3_GRAD_CLIP,
    "--dropout": config.PART3_DROPOUT,
    "--n-layer": config.PART3_N_LAYER,
    "--n-head": config.PART3_N_HEAD,
    "--n-embd": config.PART3_N_EMBD,
    "--eval-every": config.PART3_EVAL_EVERY,
    "--log-every": config.PART3_LOG_EVERY,
    "--save-every": config.PART3_SAVE_EVERY,
}


def run_dir(subset: str, seed: int, runs_dir: Path | None = None) -> Path:
    root = Path(runs_dir) if runs_dir is not None else config.RUNS_DIR
    return root / subset / f"seed{seed}"


def expected_tokens(
    max_tokens: int, block_size: int, batch_size: int, grad_accum: int
) -> int:
    tokens_per_step = block_size * batch_size * grad_accum
    return (max_tokens // tokens_per_step) * tokens_per_step


def is_complete(out_dir: Path, minimum_tokens: int | None = None) -> bool:
    summary_path = Path(out_dir) / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict) or "step" not in summary:
        return False
    try:
        tokens_seen = int(summary["tokens_seen"])
    except (KeyError, TypeError, ValueError):
        return False
    if minimum_tokens is not None and tokens_seen < minimum_tokens:
        return False
    return True


def should_resume(out_dir: Path) -> bool:
    out_dir = Path(out_dir)
    return (out_dir / "ckpt_last.pt").exists() and not is_complete(out_dir)


def build_train_command(
    subset: str,
    seed: int,
    max_tokens: int,
    python: str | None = None,
    extra: tuple[str, ...] = (),
    out_dir: Path | None = None,
    settings: dict[str, object] | None = None,
) -> list[str]:
    command = [
        python or sys.executable,
        "src/train.py",
        "--subset",
        subset,
        "--seed",
        str(seed),
        "--max-tokens",
        str(max_tokens),
    ]
    if out_dir is not None:
        command.extend(["--out-dir", str(out_dir)])
    if settings:
        for flag, value in settings.items():
            command.extend([flag, str(value)])
    command.extend(extra)
    return command


def _write_schedule(path: Path, schedule: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.PART3_SEEDS))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--max-tokens", type=int, default=config.PART3_MAX_TOKENS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--force", action="store_true", help="rerun completed pairs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    runs_root = Path(args.runs_dir).resolve()
    required_tokens = expected_tokens(
        args.max_tokens,
        config.PART3_BLOCK_SIZE,
        config.PART3_BATCH_SIZE,
        config.PART3_GRAD_ACCUM,
    )
    schedule = []

    for subset in args.subsets:
        for seed in args.seeds:
            out_dir = run_dir(subset, seed, runs_root)
            if is_complete(out_dir, required_tokens) and not args.force:
                status = "complete"
                command = None
            else:
                extra = () if args.force or not should_resume(out_dir) else ("--resume",)
                status = "resume" if extra else "run"
                command = build_train_command(
                    subset,
                    seed,
                    args.max_tokens,
                    python=args.python,
                    extra=extra,
                    out_dir=out_dir,
                    settings=TRAIN_SETTINGS,
                )
            schedule.append(
                {
                    "subset": subset,
                    "seed": seed,
                    "out_dir": str(out_dir),
                    "status": status,
                    "command": command,
                }
            )

    schedule_path = runs_root / "schedule.json"
    if args.dry_run:
        for item in schedule:
            if item["command"] is None:
                print(f"SKIP {item['subset']} seed={item['seed']} (complete)")
            else:
                print("RUN " + " ".join(item["command"]))
        return

    _write_schedule(schedule_path, schedule)
    print(f"schedule: {schedule_path}", flush=True)
    for item in schedule:
        if item["status"] == "complete":
            print(f"SKIP {item['subset']} seed={item['seed']} (complete)", flush=True)
            continue
        command = item["command"]
        assert command is not None
        print("RUN " + " ".join(command), flush=True)
        item["status"] = "running"
        _write_schedule(schedule_path, schedule)
        try:
            subprocess.run(command, cwd=config.ROOT, check=True)
        except subprocess.CalledProcessError:
            item["status"] = "failed"
            _write_schedule(schedule_path, schedule)
            print(f"FAILED {item['subset']} seed={item['seed']}", flush=True)
            raise
        item["status"] = "complete"
        _write_schedule(schedule_path, schedule)
        print(f"DONE {item['subset']} seed={item['seed']}", flush=True)


if __name__ == "__main__":
    main()
