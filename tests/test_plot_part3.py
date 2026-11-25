import json
from pathlib import Path

from plot_part3 import collect_runs, plot_val_curves, summarize, write_report


def make_run(root: Path) -> Path:
    run = root / "random" / "seed42"
    run.mkdir(parents=True)
    rows = [
        {"step": 1, "tokens": 16384, "train_loss": 4.0, "val_loss": 3.8, "val_ppl": 44.7},
        {"step": 2, "tokens": 32768, "train_loss": 3.5, "val_loss": 3.4, "val_ppl": 30.0},
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "subset": "random",
                "seed": 42,
                "final_val_loss": 3.4,
                "final_val_ppl": 30.0,
                "best_val_loss": 3.4,
                "tokens_seen": 32768,
                "tokens_per_s": 10000.0,
                "wall_time_s": 3.3,
                "peak_mem_mb": 512.0,
            }
        )
    )
    return run


def test_collect_and_summarize(tmp_path: Path):
    make_run(tmp_path)
    runs = collect_runs(tmp_path)
    assert len(runs) == 1
    report = summarize(runs)
    assert report["random"]["runs"] == 1
    assert report["random"]["mean_final_val_loss"] == 3.4
    assert report["random"]["std_final_val_loss"] == 0.0


def test_malformed_summary_is_not_fatal(tmp_path: Path):
    run = make_run(tmp_path)
    (run / "summary.json").write_text("{", encoding="utf-8")
    runs = collect_runs(tmp_path)
    assert runs[0]["summary"] == {}
    assert summarize(runs) == {}


def test_write_report_creates_json(tmp_path: Path):
    make_run(tmp_path)
    runs = collect_runs(tmp_path)
    output = tmp_path / "reports" / "part3_report.json"
    write_report(runs, output)
    report = json.loads(output.read_text())
    assert report["random"]["runs"] == 1


def test_plot_val_curves_creates_png(tmp_path: Path):
    make_run(tmp_path)
    runs = collect_runs(tmp_path)
    output = tmp_path / "figures" / "curves.png"
    plot_val_curves(runs, output)
    assert output.exists()
    assert output.stat().st_size > 0
