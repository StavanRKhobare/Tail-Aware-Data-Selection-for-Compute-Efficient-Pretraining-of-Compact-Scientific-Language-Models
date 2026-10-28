from pytest import approx

from quality import compute_signals, dup_ngram_char_frac, penalty, quality_score

CLEAN = (
    "We study the problem of learning sparse representations for graph data. "
    "Existing methods require dense adjacency matrices, which limits their scale. "
    "We propose a spectral filtering approach that works directly on edge lists and "
    "show that it improves node classification accuracy on five public benchmarks "
    "while using far less memory than prior work. Our analysis explains when the "
    "approximation is accurate and when it fails, and we release our code to support "
    "further research on scalable graph learning methods for practitioners and researchers."
)


def test_clean_abstract_scores_one():
    assert quality_score(compute_signals(CLEAN)) == 1.0


def test_math_is_not_counted_as_symbols():
    math_heavy = CLEAN + " We prove a bound of $O(\\sqrt{T} \\log T)$ with $\\|x\\|_2^2 \\le \\epsilon$."
    s = compute_signals(math_heavy)
    assert s["math_frac"] > 0
    assert s["symbol_frac"] == 0
    assert s["latex_cmds"] == 0


def test_latex_outside_math_is_counted():
    s = compute_signals(CLEAN + " As shown in \\citep{smith2020} and \\textit{prior work}.")
    assert s["latex_cmds"] == 2


def test_acronyms_are_not_foreign_words():
    s = compute_signals("We predict LOS and DAS events. " + CLEAN)
    assert s["foreign_frac"] == 0
    s = compute_signals("Nous proposons une méthode pour les données avec des graphes et nous " + CLEAN)
    assert s["foreign_frac"] > 0


def test_repetition_lowers_score():
    repetitive = " ".join(["the unmanned aerial vehicle navigation"] * 30)
    assert dup_ngram_char_frac(repetitive.split(), 3) > 0.9
    assert quality_score(compute_signals(repetitive)) < 0.5


def test_short_abstract_penalized():
    short = "We show that the problem is hard. We also give an algorithm for it."
    assert quality_score(compute_signals(short)) < 0.5


def test_penalty_ramp():
    assert penalty(0.10, "high", 0.15, 0.35) == 0.0
    assert penalty(0.25, "high", 0.15, 0.35) == approx(0.5)
    assert penalty(0.90, "high", 0.15, 0.35) == 1.0
    assert penalty(100, "low", 80, 40) == 0.0
    assert penalty(60, "low", 80, 40) == 0.5
    assert penalty(10, "low", 80, 40) == 1.0
