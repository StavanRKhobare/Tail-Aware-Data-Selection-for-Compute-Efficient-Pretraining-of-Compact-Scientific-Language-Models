"""Budgeted document sampling, with optional topic-aware allocation.

All strategies share one mechanism:
  1. order the candidate pool by a fixed random key per document (the same key
     in every strategy, so two strategies differ only where their rules differ);
  2. if topic-aware, split the token budget across topics with weights
     tokens_c ** alpha (water-filling: a topic never gets more than it has,
     and what it cannot use is shared among the others);
  3. take documents in shuffled order until the (per-topic) budget is used.
"""
import numpy as np
import pandas as pd


def temperature_weights(sizes: pd.Series, alpha: float) -> pd.Series:
    """alpha = 1 keeps the natural mix, alpha = 0 is uniform over topics."""
    w = sizes.astype(float) ** alpha
    return w / w.sum()


def allocate(available: pd.Series, weights: pd.Series, budget: float) -> pd.Series:
    """Water-filling split of `budget` across topics, capped by `available`."""
    alloc = pd.Series(0.0, index=available.index)
    active = available.index[available > 0]
    remaining = float(budget)
    while remaining > 1e-9 and len(active):
        share = remaining * weights[active] / weights[active].sum()
        room = available[active] - alloc[active]
        capped = (share >= room).to_numpy()
        if capped.any():
            alloc[active[capped]] = available[active[capped]]
            remaining -= float(room[capped].sum())
            active = active[~capped]
        else:
            alloc[active] += share
            remaining = 0.0
    return alloc


def random_keys(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def select(pool: pd.DataFrame, budget: int, topic: str | None = None,
           alpha: float = 1.0, key: str = "rand") -> pd.DataFrame:
    """Pick documents from `pool` up to `budget` tokens.

    `pool` needs a `tokens` column and a random `key` column (see random_keys);
    documents are taken in key order. Returns the chosen rows. The total never
    exceeds the budget and falls short by less than one document.
    """
    pool = pool.sort_values(key, kind="stable")
    if topic is None:
        return pool[pool["tokens"].cumsum() <= budget]

    available = pool.groupby(topic)["tokens"].sum()
    alloc = allocate(available, temperature_weights(available, alpha), budget)
    chosen = pool.groupby(topic)["tokens"].cumsum() <= pool[topic].map(alloc)
    # Per-topic rounding leaves a small remainder: top it up in shuffled order.
    room = budget - pool.loc[chosen, "tokens"].sum()
    rest = pool.loc[~chosen, "tokens"]
    chosen.loc[rest.index[rest.cumsum() <= room]] = True
    return pool[chosen]
