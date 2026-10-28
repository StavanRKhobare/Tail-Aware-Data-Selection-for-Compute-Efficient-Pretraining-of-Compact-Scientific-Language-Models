import numpy as np

from minhash import candidate_pairs, components, signatures, similarity

BASE = (
    "we propose a novel method for learning sparse representations of graphs using spectral "
    "filters and show that it improves accuracy on node classification benchmarks while reducing "
    "memory by a large factor compared to prior work on scalable graph neural networks and we "
    "analyse when the approximation holds in theory and in practice across many datasets"
)
OTHER = (
    "an entirely different abstract about robot grasping with tactile sensors and reinforcement "
    "learning in cluttered household scenes where objects are deformable and partially occluded"
)


def test_near_duplicate_found_and_unrelated_not():
    texts = [BASE, BASE.replace("novel", "new"), OTHER, "tiny"]
    sigs = signatures(texts)
    pairs = candidate_pairs(sigs, 16)
    sim = similarity(sigs, pairs)
    found = {tuple(p) for p, s in zip(pairs.tolist(), sim) if s >= 0.8}
    assert found == {(0, 1)}


def test_signatures_are_deterministic():
    assert np.array_equal(signatures([BASE, OTHER], seed=3), signatures([BASE, OTHER], seed=3))


def test_components_root_is_lowest_index():
    root = components(5, np.array([[3, 4], [1, 4]]))
    assert root.tolist() == [0, 1, 2, 1, 1]
