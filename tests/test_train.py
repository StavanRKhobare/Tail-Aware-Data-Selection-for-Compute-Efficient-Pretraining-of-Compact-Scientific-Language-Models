import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model import GPT, GPTConfig
from train import (
    build_scheduler,
    load_checkpoint,
    resume_mismatches,
    truncate_metrics,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA RNG state")
def test_load_checkpoint_keeps_rng_state_on_cpu(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        path,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["torch_rng_state"].device.type == "cpu"
    assert all(state.device.type == "cpu" for state in checkpoint["cuda_rng_state"])


def test_truncate_metrics_removes_steps_after_checkpoint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"step": step}) for step in (0, 10, 20, 30)) + "\n"
    )
    truncate_metrics(path, 10)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 10]


def test_truncate_metrics_skips_a_partial_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 0}) + "\n" + json.dumps({"step": 10}) + "\n" + '{"step": 2',
        encoding="utf-8",
    )
    truncate_metrics(path, 10)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 10]


def test_resume_mismatches_flags_changed_schedule_settings():
    saved = {"subset": "random", "batch_size": 32, "lr": 6e-4, "max_steps": 2}
    changed = {
        "subset": "random",
        "batch_size": 16,
        "lr": 6e-4,
        "max_steps": 3,
        "resume": True,
    }
    mismatches = resume_mismatches(saved, changed)
    assert len(mismatches) == 1
    assert "batch_size" in mismatches[0]
    assert (
        resume_mismatches(
            saved,
            {"subset": "random", "batch_size": 32, "lr": 6e-4, "max_steps": 3},
        )
        == []
    )


def test_scheduler_starts_nonzero_and_ends_at_floor():
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_scheduler(
        optimizer, max_steps=100, warmup_frac=0.02, min_lr_ratio=0.1
    )
    assert scheduler.get_last_lr()[0] > 0.0
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1, rel=1e-6)


def make_fake_data(root: Path):
    data_dir = root / "data"
    run_dir = data_dir / "smoke"
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 64, size=2000, dtype=np.uint16).tofile(run_dir / "train.bin")
    rng.integers(0, 64, size=500, dtype=np.uint16).tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(
        json.dumps({"vocab_size": 64, "eot_id": 0, "pad_id": 1})
    )
    return data_dir


def tiny_cmd(data_dir: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        "src/train.py",
        "--subset",
        "smoke",
        "--seed",
        "42",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--batch-size",
        "2",
        "--block-size",
        "8",
        "--n-layer",
        "1",
        "--n-head",
        "2",
        "--n-embd",
        "32",
        "--device",
        "cpu",
        "--no-amp",
        "--eval-every",
        "1",
        "--log-every",
        "1",
        "--save-every",
        "1",
        *extra,
    ]


def test_train_cli_writes_checkpoints_and_metrics(tmp_path):
    data_dir = make_fake_data(tmp_path)
    out_dir = tmp_path / "run"
    cmd = tiny_cmd(data_dir, out_dir, "--max-steps", "3")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (out_dir / "ckpt_last.pt").exists()
    assert (out_dir / "model_last.pt").exists()
    assert (out_dir / "metrics.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["step"] == 3
    assert summary["tokens_seen"] == 3 * 2 * 8
    assert summary["final_val_loss"] is not None


def test_train_cli_resumes_without_duplicate_metrics(tmp_path):
    data_dir = make_fake_data(tmp_path)
    out_dir = tmp_path / "run"
    first = subprocess.run(
        tiny_cmd(data_dir, out_dir, "--max-steps", "2"),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr
    metrics = out_dir / "metrics.jsonl"
    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 99, "val_loss": 1.0}) + "\n")

    resumed = subprocess.run(
        tiny_cmd(data_dir, out_dir, "--max-steps", "3", "--resume"),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["step"] == 3
    assert summary["tokens_seen"] == 3 * 2 * 8
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert max(row["step"] for row in rows) == 3
    assert sum(row["step"] == 0 for row in rows) == 1
    assert summary["wall_time_s"] > 0
