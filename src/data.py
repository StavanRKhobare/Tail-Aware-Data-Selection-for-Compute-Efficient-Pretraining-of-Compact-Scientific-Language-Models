"""Memmapped token windows for Part 3 training.

`train.bin` and `val.bin` are flat uint16 arrays in the nanoGPT format: every
document's tokens followed by one end-of-text token. Windows are non-overlapping
blocks of `block_size + 1` tokens; the model receives the first `block_size`
tokens and predicts the next token at every position.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


class TokenWindows:
    def __init__(self, path: str | Path, block_size: int):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.path = Path(path)
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_windows = (len(self.data) - 1) // block_size
        if self.n_windows < 1:
            raise ValueError(f"{self.path} is too short for block_size={block_size}")

    def get_batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        starts = indices * self.block_size
        x = np.stack([self.data[start : start + self.block_size] for start in starts])
        y = np.stack([self.data[start + 1 : start + 1 + self.block_size] for start in starts])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    def val_batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.arange(self.n_windows, dtype=np.int64)
        for start in range(0, len(indices), batch_size):
            yield self.get_batch(indices[start : start + batch_size])


class TokenStream:
    """Infinite stream of shuffled, non-overlapping training windows.

    Each epoch shuffles every window once and drops only a remainder smaller
    than one batch. The epoch permutation depends only on `seed`, so a resumed
    run can skip whole optimizer steps and reproduce the next batch exactly.
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        batch_size: int,
        grad_accum: int,
        seed: int,
    ):
        if batch_size < 1 or grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be positive")
        self.windows = TokenWindows(path, block_size)
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.rng = np.random.default_rng(seed)
        self._order: np.ndarray | None = None
        self._position = 0

    def _new_epoch(self) -> None:
        if self.windows.n_windows < self.batch_size:
            raise ValueError("the token file is shorter than one batch")
        self._order = self.rng.permutation(self.windows.n_windows)
        self._position = 0

    def _skip_micro_batches(self, count: int) -> None:
        while count > 0:
            if self._order is None or self._position >= self.windows.n_windows:
                self._new_epoch()
            remaining = (self.windows.n_windows - self._position) // self.batch_size
            if remaining == 0:
                self._new_epoch()
                continue
            take = min(count, remaining)
            self._position += take * self.batch_size
            count -= take

    def skip_steps(self, steps: int) -> None:
        self._skip_micro_batches(steps * self.grad_accum)

    def next_micro_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._order is None or self._position + self.batch_size > self.windows.n_windows:
            self._new_epoch()
        assert self._order is not None
        indices = self._order[self._position : self._position + self.batch_size]
        self._position += self.batch_size
        return self.windows.get_batch(indices)


def steps_for_tokens(
    max_tokens: int, block_size: int, batch_size: int, grad_accum: int
) -> int:
    tokens_per_step = block_size * batch_size * grad_accum
    if tokens_per_step <= 0:
        raise ValueError("tokens per step must be positive")
    return max_tokens // tokens_per_step
