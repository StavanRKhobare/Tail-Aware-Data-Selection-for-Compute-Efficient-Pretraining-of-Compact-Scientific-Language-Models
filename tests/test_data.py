import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data import TokenStream, TokenWindows, steps_for_tokens


def write_bin(tmp_path, n=1000):
    path = tmp_path / "train.bin"
    np.arange(n, dtype=np.uint16).tofile(path)
    return path


def test_windows_are_shifted_by_one(tmp_path):
    windows = TokenWindows(write_bin(tmp_path), block_size=8)
    x, y = windows.get_batch(np.array([0, 1]))
    assert x.tolist()[0] == list(range(8))
    assert y.tolist()[0] == list(range(1, 9))
    assert x.tolist()[1] == list(range(8, 16))
    assert y.tolist()[1] == list(range(9, 17))


def test_validation_batches_cover_all_windows(tmp_path):
    windows = TokenWindows(write_bin(tmp_path), block_size=8)
    batches = list(windows.val_batches(batch_size=32))
    assert sum(x.shape[0] for x, _ in batches) == windows.n_windows


def test_steps_for_tokens():
    assert steps_for_tokens(1000, 8, 2, 2) == 31


def test_stream_skip_matches_advancing(tmp_path):
    path = write_bin(tmp_path)
    a = TokenStream(path, block_size=8, batch_size=2, grad_accum=1, seed=7)
    for _ in range(5):
        a.next_micro_batch()
    expected = a.next_micro_batch()

    b = TokenStream(path, block_size=8, batch_size=2, grad_accum=1, seed=7)
    b.skip_steps(5)
    actual = b.next_micro_batch()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
