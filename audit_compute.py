"""Read-only extraction of existing Part 3 compute measurements.

No runtime is estimated without a label. The only extrapolations are simple
throughput calculations from measured tokens/second, plus parameter counts
computed by instantiating the existing model class. No model is trained.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, GPTConfig  # noqa: E402

STRATEGIES = ("random", "quality", "diversity", "tail_aware", "tail_only")
SEEDS = (42, 43)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tag", default=None, help="Prefix for output filenames.")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    prefix = f"{args.tag}_" if args.tag else ""

    def out(name: str) -> Path:
        return ROOT / "reports" / f"{prefix}{name}"

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_info = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_mb": props.total_memory / 1024**2,
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    else:
        device_info = {"name": "cpu", "total_memory_mb": None, "torch": torch.__version__}
    print(json.dumps(device_info, indent=2))

    runs = []
    for strategy in STRATEGIES:
        for seed in seeds:
            run_dir = ROOT / "runs" / strategy / f"seed{seed}"
            cfg = read_json(run_dir / "config.json")
            summary = read_json(run_dir / "summary.json")
            records = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            evals = [r for r in records if r.get("val_loss") is not None]
            last = evals[-1]
            wall = summary["wall_time_s"]
            train_elapsed = last["elapsed_s"]
            runs.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "device": summary["device"],
                    "amp_dtype": summary["amp_dtype"],
                    "params": summary["model_params"],
                    "steps": summary["step"],
                    "planned_tokens": summary["planned_tokens"],
                    "tokens_seen": summary["tokens_seen"],
                    "final_val_loss": summary["final_val_loss"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_step": summary["best_step"],
                    "wall_time_s": wall,
                    "wall_time_min": wall / 60,
                    "train_segment_elapsed_s": train_elapsed,
                    "eval_overhead_s": wall - train_elapsed,
                    "tokens_per_s_summary": summary["tokens_per_s"],
                    "tokens_per_s_train_segment": summary["tokens_seen"] / train_elapsed,
                    "peak_mem_mb": summary["peak_mem_mb"],
                    "peak_mem_pct_gpu": (
                        100 * summary["peak_mem_mb"] / device_info["total_memory_mb"]
                        if device_info["total_memory_mb"]
                        else None
                    ),
                    "s_per_million_tokens": wall / (summary["tokens_seen"] / 1e6),
                    "eval_points": len(evals),
                    "n_metrics_rows": len(records),
                    "train_bin": summary["train_bin"],
                    "config": cfg,
                }
            )

    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "config"} for row in runs])
    frame.to_csv(out("compute_efficiency_runs.csv"), index=False)

    # Verify the equal-compute design rather than assuming it.
    configs = []
    for row in runs:
        cfg = row["config"]
        configs.append(
            {
                "strategy": row["strategy"],
                "seed": row["seed"],
                "model": cfg["model"],
                "train": {
                    k: cfg["train"][k]
                    for k in (
                        "max_tokens", "block_size", "batch_size", "grad_accum", "lr",
                        "min_lr_ratio", "warmup_frac", "weight_decay", "grad_clip",
                        "eval_every", "log_every", "save_every",
                    )
                },
                "data": cfg["data"],
                "max_steps": cfg["max_steps"],
                "tokens_per_step": cfg["tokens_per_step"],
                "planned_tokens": cfg["planned_tokens"],
            }
        )
    first = configs[0]
    mismatches = []
    for cfg in configs[1:]:
        for key in ("model", "train", "max_steps", "tokens_per_step", "planned_tokens"):
            if cfg[key] != first[key]:
                mismatches.append({"strategy": cfg["strategy"], "seed": cfg["seed"], "field": key})

    strategy_summary = (
        frame.groupby("strategy")
        .agg(
            runs=("seed", "count"),
            steps=("steps", "first"),
            tokens_seen=("tokens_seen", "first"),
            wall_time_s_mean=("wall_time_s", "mean"),
            wall_time_s_std=("wall_time_s", "std"),
            tokens_per_s_mean=("tokens_per_s_summary", "mean"),
            tokens_per_s_std=("tokens_per_s_summary", "std"),
            peak_mem_mb_mean=("peak_mem_mb", "mean"),
            eval_overhead_s_mean=("eval_overhead_s", "mean"),
        )
        .reindex(STRATEGIES)
        .reset_index()
    )
    strategy_summary.to_csv(
        out("compute_efficiency_strategy_summary.csv"), index=False
    )

    measured_tps = float(frame["tokens_per_s_summary"].mean())
    measured_s_per_run = float(frame["wall_time_s"].mean())
    tokens_per_run = int(frame["tokens_seen"].iloc[0])

    # 120M-token extension: simple throughput calculation from the measured
    # 60M runs. This is an arithmetic estimate, not a measured 120M run.
    tokens_120m = 120_000_000
    wall_120m_s = tokens_120m / measured_tps
    wall_60m_s = tokens_per_run / measured_tps

    # Future model counts, computed by instantiating the existing class.
    model_candidates = []
    for layers, heads, width in ((6, 6, 384), (8, 8, 512), (9, 8, 512), (10, 8, 512)):
        params = GPT(GPTConfig(n_layer=layers, n_head=heads, n_embd=width)).num_parameters()
        ratio = params / frame["params"].iloc[0]
        model_candidates.append(
            {
                "n_layer": layers,
                "n_head": heads,
                "n_embd": width,
                "block_size": 512,
                "params": params,
                "params_millions": params / 1e6,
                "param_ratio_vs_current": ratio,
                "est_tokens_per_s_linear_param_scaling": measured_tps / ratio,
                "est_60m_wall_min_linear_param_scaling": (60e6 / (measured_tps / ratio)) / 60,
                "est_optimizer_param_state_mb": params * 16 / 1e6,
            }
        )

    estimates = {
        "measured": {
            "runs": len(frame),
            "tokens_per_run": tokens_per_run,
            "steps_per_run": int(frame["steps"].iloc[0]),
            "mean_wall_time_s": measured_s_per_run,
            "mean_wall_time_min": measured_s_per_run / 60,
            "mean_tokens_per_s": measured_tps,
            "mean_peak_allocated_mb": float(frame["peak_mem_mb"].mean()),
            "mean_eval_overhead_s": float(frame["eval_overhead_s"].mean()),
            "wall_60m_arithmetic_s_at_mean_throughput": wall_60m_s,
        },
        "longer_training_120m": {
            "label": "arithmetic estimate from measured mean throughput; no 120M run exists",
            "wall_per_run_s": wall_120m_s,
            "wall_per_run_min": wall_120m_s / 60,
            "wall_10_runs_h": 10 * wall_120m_s / 3600,
            "wall_15_runs_h": 15 * wall_120m_s / 3600,
            "wall_25_runs_h": 25 * wall_120m_s / 3600,
        },
        "model_scaling": {
            "label": "parameter counts are exact; runtime and memory extrapolations are "
            "linear-scaling estimates and must be verified by a short pilot",
            "candidates": model_candidates,
            "recommended_candidate": next(
                c for c in model_candidates if c["n_layer"] == 9 and c["n_head"] == 8 and c["n_embd"] == 512
            ),
        },
    }
    (out("compute_efficiency_audit.json")).write_text(
        json.dumps(
            {
                "device": device_info,
                "equal_compute_mismatches": mismatches,
                "missing_instrumentation": [
                    "separate training and validation wall time per run",
                    "GPU utilization, power draw, and energy",
                    "torch.cuda.max_memory_reserved in addition to allocated",
                    "per-step latency distribution (only 10-step log rows exist)",
                    "CPU time, host RAM, and dataloader time",
                    "exact number of optimization steps in the final partial evaluation window",
                ],
                "estimates": estimates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(model_candidates).to_csv(
        out("compute_efficiency_model_estimates.csv"), index=False
    )

    print("\nRUNS")
    print(frame.to_string(index=False))
    print("\nSTRATEGY SUMMARY")
    print(strategy_summary.to_string(index=False))
    print("\nMISMATCHES", mismatches)
    print("\nESTIMATES")
    print(json.dumps(estimates, indent=2))
    print(f"\nwrote reports/{prefix}compute_efficiency_*.{{json,csv}}")


if __name__ == "__main__":
    main()
