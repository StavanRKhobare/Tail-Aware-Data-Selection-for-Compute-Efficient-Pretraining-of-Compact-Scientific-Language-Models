from pathlib import Path

from train_all import (
    build_train_command,
    expected_tokens,
    is_complete,
    should_resume,
)


def test_build_train_command_uses_fixed_budget():
    cmd = build_train_command("random", 42, 60_000_000, python="python")
    assert cmd[:2] == ["python", "src/train.py"]
    assert "--subset" in cmd and "random" in cmd
    assert "--seed" in cmd and "42" in cmd
    assert "--max-tokens" in cmd and "60000000" in cmd


def test_build_train_command_passes_frozen_settings():
    cmd = build_train_command(
        "random", 42, 60_000_000, python="python", settings={"--lr": 6e-4}
    )
    assert cmd[cmd.index("--lr") + 1] == "0.0006"
    assert "--block-size" not in cmd


def test_expected_tokens_rounds_to_whole_steps():
    assert expected_tokens(60_000_000, 512, 32, 1) == 59_998_208


def test_should_resume_partial_run(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "ckpt_last.pt").write_bytes(b"x")
    assert should_resume(out) is True
    assert is_complete(out) is False


def test_is_complete_needs_a_valid_summary(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "summary.json").write_text(
        '{"step": 3662, "tokens_seen": 59998208}', encoding="utf-8"
    )
    assert is_complete(out) is True
    assert should_resume(out) is False

    (out / "summary.json").write_text("{", encoding="utf-8")
    assert is_complete(out) is False


def test_is_complete_rejects_a_shorter_budget(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "summary.json").write_text(
        '{"step": 1831, "tokens_seen": 29999104}', encoding="utf-8"
    )
    assert is_complete(out, minimum_tokens=59_998_208) is False
