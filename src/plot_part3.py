"""Aggregate Part 3 runs into a JSON report and a validation-loss figure."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SUBSET_ORDER = ("random", "quality", "diversity", "tail_aware", "tail_only")
COLORS = {name: f"C{i}" for i, name in enumerate(SUBSET_ORDER)}


def collect_runs(runs_dir: str | Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(Path(runs_dir).glob("*/seed*/metrics.jsonl")):
        run_path = metrics_path.parent
        subset = run_path.parent.name
        seed = int(run_path.name.removeprefix("seed"))
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary_path = run_path / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                summary = loaded
        runs.append(
            {
                "subset": subset,
                "seed": seed,
                "dir": str(run_path),
                "metrics": rows,
                "summary": summary,
            }
        )
    return runs


def _values(runs: list[dict], key: str) -> list[float]:
    values = []
    for run in runs:
        value = run.get("summary", {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.stdev(values) if len(values) > 1 else 0.0, 6)


def summarize(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        if run.get("summary") and run["summary"].get("final_val_loss") is not None:
            grouped.setdefault(run["subset"], []).append(run)

    report = {}
    for subset in SUBSET_ORDER:
        group = grouped.get(subset, [])
        if not group:
            continue
        finals = _values(group, "final_val_loss")
        ppls = _values(group, "final_val_ppl")
        tokens = _values(group, "tokens_seen")
        throughputs = _values(group, "tokens_per_s")
        walls = _values(group, "wall_time_s")
        peaks = _values(group, "peak_mem_mb")
        report[subset] = {
            "runs": len(group),
            "seeds": sorted(run["seed"] for run in group),
            "mean_tokens_seen": _mean(tokens),
            "mean_final_val_loss": _mean(finals),
            "std_final_val_loss": _std(finals),
            "mean_final_val_ppl": _mean(ppls),
            "mean_tokens_per_s": _mean(throughputs),
            "mean_wall_time_s": _mean(walls),
            "mean_peak_mem_mb": _mean(peaks),
        }
    return report


def write_report(runs: list[dict], out_path: str | Path) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = summarize(runs)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def plot_val_curves(runs: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        rows = [row for row in run["metrics"] if row.get("val_loss") is not None]
        if not rows:
            continue
        x = [row["tokens"] for row in rows]
        y = [row["val_loss"] for row in rows]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.2,
            alpha=0.85,
            color=COLORS.get(run["subset"], "C7"),
            label=f"{run['subset']} seed{run['seed']}",
        )
    ax.set_xlabel("training tokens")
    ax.set_ylabel("validation loss")
    ax.set_title("Part 3 validation loss by training subset")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--out-json", type=Path, default=config.REPORTS_DIR / "part3_report.json")
    parser.add_argument("--out-fig", type=Path, default=config.FIG_DIR / "part3_val_loss.png")
    args = parser.parse_args(argv)

    runs = collect_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.runs_dir}")
    report = write_report(runs, args.out_json)
    plot_val_curves(runs, args.out_fig)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out_json} and {args.out_fig}")


if __name__ == "__main__":
    main()
