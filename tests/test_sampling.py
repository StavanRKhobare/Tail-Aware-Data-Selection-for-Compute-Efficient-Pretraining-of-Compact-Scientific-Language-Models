import numpy as np
import pandas as pd
import pytest

from sampling import allocate, random_keys, select, temperature_weights


def test_temperature_weights():
    sizes = pd.Series({"a": 100.0, "b": 1.0})
    assert temperature_weights(sizes, 1.0)["a"] == pytest.approx(100 / 101)
    assert temperature_weights(sizes, 0.0)["a"] == pytest.approx(0.5)
    assert temperature_weights(sizes, 0.5)["a"] == pytest.approx(10 / 11)


def test_allocate_caps_and_redistributes():
    available = pd.Series({"head": 1000.0, "mid": 100.0, "tail": 10.0})
    alloc = allocate(available, pd.Series({"head": 1.0, "mid": 1.0, "tail": 1.0}), 300)
    assert alloc["tail"] == 10  # capped at what it has
    assert alloc["mid"] == 100
    assert alloc["head"] == pytest.approx(190)  # receives the leftover
    assert alloc.sum() == pytest.approx(300)


def test_allocate_budget_larger_than_pool():
    available = pd.Series({"a": 5.0, "b": 7.0})
    assert allocate(available, pd.Series({"a": 0.5, "b": 0.5}), 100).tolist() == [5, 7]


def make_pool(seed=0):
    rng = np.random.default_rng(seed)
    topics = ["head"] * 900 + ["mid"] * 90 + ["tail"] * 10
    return pd.DataFrame({"id": [f"d{i}" for i in range(1000)], "topic": topics,
                         "tokens": rng.integers(100, 300, size=1000), "rand": random_keys(1000, seed)})


@pytest.mark.parametrize("topic,alpha", [(None, 1.0), ("topic", 0.5), ("topic", 0.0)])
def test_select_respects_budget(topic, alpha):
    pool = make_pool()
    sub = select(pool, 20_000, topic=topic, alpha=alpha)
    assert sub["tokens"].sum() <= 20_000
    assert 20_000 - sub["tokens"].sum() < pool["tokens"].max()
    assert sub["id"].is_unique


def test_select_is_deterministic():
    pool = make_pool()
    a = select(pool, 20_000, topic="topic", alpha=0.5)
    b = select(pool, 20_000, topic="topic", alpha=0.5)
    assert a["id"].tolist() == b["id"].tolist()


def test_tail_aware_upweights_tail():
    pool = make_pool()
    natural = select(pool, 20_000)
    tail = select(pool, 20_000, topic="topic", alpha=0.5)
    share = lambda s: s.loc[s["topic"] == "tail", "tokens"].sum() / s["tokens"].sum()
    assert share(tail) > 3 * share(natural)
    # sqrt-weighted share of tail is ~7%: at this budget that is less than the tail holds
    assert 0 < (tail["topic"] == "tail").sum() < 10


def test_tail_fully_taken_when_its_share_exceeds_it():
    pool = make_pool()
    tail = select(pool, 60_000, topic="topic", alpha=0.5)
    assert (tail["topic"] == "tail").sum() == 10


def test_common_random_numbers_keep_subsets_aligned():
    """Filtering 5% of the pool should change only about 5% of the chosen documents."""
    pool = make_pool()
    full = select(pool, 50_000)
    filtered = select(pool[pool.index % 20 != 0], 50_000)
    overlap = len(set(full["id"]) & set(filtered["id"])) / len(set(full["id"]) | set(filtered["id"]))
    assert overlap > 0.85
