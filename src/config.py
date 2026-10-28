from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

HF_DATASET = "librarian-bots/arxiv-metadata-snapshot"
TARGET_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI", "stat.ML"}

SEED = 42
SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}

# Text format used for training: "title\n\nabstract" or abstract only.
# Decide once and keep it identical across all selection strategies.
INCLUDE_TITLE = True

# Starting thresholds. Tune them after looking at the inspect_corpus.py output.
MIN_WORDS = 20
MAX_WORDS = 800

# ---------------------------------------------------------------------------
# Part 2: data selection and tokenization
# ---------------------------------------------------------------------------
SELECTION_DIR = ROOT / "data" / "selection"   # per-doc scores, topics, token counts
SUBSETS_DIR = ROOT / "data" / "subsets"       # tokenized .bin files for Part 3
TOKENIZER_DIR = ROOT / "tokenizer"            # committed: the fixed tokenizer
MANIFEST_DIR = ROOT / "manifests"             # committed: ids of each subset

# Byte-level BPE. 16k keeps the embedding table small for a 10-20M param model.
VOCAB_SIZE = 16_384
EOT_TOKEN = "<|endoftext|>"   # appended after every document
PAD_TOKEN = "<|pad|>"

# Topic groups by number of train papers per primary category.
HEAD_MIN_DOCS = 10_000   # head: >= 10,000
MID_MIN_DOCS = 1_000     # mid: 1,000 to 9,999; tail: < 1,000

# Text clusters for diversity-aware selection (fit on train only):
# hierarchical k-means, 1,000 fine clusters grouped into 100 themes.
N_FINE_CLUSTERS = 1_000
N_CLUSTERS = 100

# Every subset gets the same number of tokens (document tokens + one EOT each).
TOKEN_BUDGET = 30_000_000

# Temperature for tail-aware sampling over primary categories:
# weight_c ~ tokens_c ** alpha. 1 = natural mix, 0 = uniform, 0.5 = in between.
TAIL_ALPHA = 0.5
# Diversity-aware sampling uses equal weight per text theme (alpha = 0).
DIVERSITY_ALPHA = 0.0

# Quality filter: keep documents with quality_score >= QUALITY_MIN that are not
# near-duplicates. 0.5 = drop only documents with a clear issue (see quality.py).
QUALITY_MIN = 0.5

# Near-duplicate detection (MinHash on word 5-grams of the abstract).
# Pairs at 0.5-0.8 were read by hand: revised / extended versions of the same
# paper under a different id. Below 0.5 they are mostly shared templates.
MINHASH_PERMS = 128
MINHASH_BANDS = 32          # 32 bands x 4 rows -> candidate threshold ~0.42
DUP_JACCARD = 0.5           # verified estimated Jaccard to call two abstracts the same work


def ensure_dirs():
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIG_DIR,
              SELECTION_DIR, SUBSETS_DIR, TOKENIZER_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)
