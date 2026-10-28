"""Per-document quality signals for scientific abstracts.

Each signal is cheap, interpretable and computed from the abstract alone.
LaTeX math ($...$, \\(...\\), \\[...\\]) and URLs are removed before the
symbol / language checks, so math-heavy abstracts are not penalized for
their math (DATASET.md limitation 6). Math share is reported separately.
"""
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$]+\$|\\\(.+?\\\)|\\\[.+?\\\]", re.S)
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|org|io|net)/\S*", re.I)
LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
WORD_RE = re.compile(r"[A-Za-z]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Ordinary prose punctuation; anything else that is not alphanumeric counts as a symbol.
PROSE_PUNCT = set(".,;:!?'\"()[]-/%")

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
# Function words of French / Spanish / Portuguese / German that are not English words.
FOREIGN_FUNCTION_WORDS = frozenset(
    "les des une dans pour nous sont avec cette ces aux est sur leur qui "
    "los las del por para una con como pero sus "
    "uma com dos das nao sao pelo pela "
    "der und das ist nicht mit eine von wir sich auf werden".split()
)

SIGNALS = (
    "n_words", "math_frac", "symbol_frac", "latex_cmds", "urls", "non_ascii_frac",
    "stopword_frac", "foreign_frac", "mean_word_len", "caps_frac",
    "dup_3gram_frac", "top_2gram_frac", "n_sentences", "mean_sentence_words",
)


# Graded penalties: (signal, direction, start, end). The penalty is 0 at `start`,
# rises linearly, and is 1 at `end`. Thresholds were set by reading abstracts at
# each signal's extremes (see PART2.md); most abstracts get no penalty at all.
PENALTIES = (
    ("n_words", "low", 80, 40),                 # very short: little content
    ("dup_3gram_frac", "high", 0.15, 0.35),     # repeated phrases
    ("top_2gram_frac", "high", 0.10, 0.20),     # one phrase dominates
    ("symbol_frac", "high", 0.005, 0.03),       # markdown / stray markup outside math
    ("latex_cmds", "high", 0, 6),               # \cite, \textit, ... outside math
    ("mean_sentence_words", "high", 40, 80),    # run-on text / missing punctuation
    ("mean_sentence_words", "low", 10, 5),      # fragments / lists
    ("stopword_frac", "low", 0.20, 0.10),       # not running prose
    ("foreign_frac", "high", 0.02, 0.06),       # non-English text
    ("caps_frac", "high", 0.12, 0.25),          # acronym soup / all caps
)


def penalty(value, direction, start, end):
    """0 inside the normal range, 1 beyond `end`, linear in between."""
    if direction == "high":
        p = (value - start) / (end - start)
    else:
        p = (start - value) / (start - end)
    return min(max(p, 0.0), 1.0)


def quality_score(sig: dict) -> float:
    """Product of (1 - penalty) over all rules: 1.0 = no issues, 0.0 = severe issue."""
    q = 1.0
    for name, direction, start, end in PENALTIES:
        q *= 1.0 - penalty(sig[name], direction, start, end)
    return q


def dup_ngram_char_frac(words, n):
    """Fraction of word characters covered by word n-grams that occur more than once (Gopher)."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    total = sum(len(w) for w in words)
    return sum(len(w) for w, c in zip(words, covered) if c) / max(total, 1)


def top_ngram_char_frac(words, n):
    """Character share of the single most frequent word n-gram (Gopher)."""
    if len(words) < n:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    return count * sum(len(w) for w in gram) / max(sum(len(w) for w in words), 1)


def compute_signals(abstract: str) -> dict:
    n_chars = max(len(abstract), 1)
    math_chars = sum(len(m) for m in MATH_RE.findall(abstract))
    prose = MATH_RE.sub(" ", abstract)
    urls = len(URL_RE.findall(prose))
    prose = URL_RE.sub(" ", prose)

    visible = [c for c in prose if not c.isspace()]
    n_visible = max(len(visible), 1)
    symbols = sum(1 for c in visible if not c.isalnum() and c not in PROSE_PUNCT)
    non_ascii = sum(1 for c in visible if ord(c) > 127)

    tokens = prose.lower().split()
    alpha_words = WORD_RE.findall(prose)
    lower_words = [w.lower() for w in alpha_words]
    n_alpha = max(len(alpha_words), 1)
    caps = sum(1 for w in alpha_words if len(w) >= 2 and w.isupper())

    sentences = [s for s in SENT_RE.split(prose.strip()) if s.strip()]
    n_words = len(abstract.split())

    return {
        "n_words": n_words,
        "math_frac": math_chars / n_chars,
        "symbol_frac": symbols / n_visible,
        "latex_cmds": len(LATEX_CMD_RE.findall(prose)),
        "urls": urls,
        "non_ascii_frac": non_ascii / n_visible,
        "stopword_frac": sum(w in STOPWORDS for w in lower_words) / n_alpha,
        # Lowercase words only, so acronyms like LOS or DAS do not count as Spanish / German.
        "foreign_frac": sum(w in FOREIGN_FUNCTION_WORDS for w in alpha_words) / n_alpha,
        "mean_word_len": sum(len(w) for w in alpha_words) / n_alpha,
        "caps_frac": caps / n_alpha,
        "dup_3gram_frac": dup_ngram_char_frac(tokens, 3),
        "top_2gram_frac": top_ngram_char_frac(tokens, 2),
        "n_sentences": len(sentences),
        "mean_sentence_words": n_words / max(len(sentences), 1),
    }
