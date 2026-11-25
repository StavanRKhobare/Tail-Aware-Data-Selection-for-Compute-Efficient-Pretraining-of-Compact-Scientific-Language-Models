"""Run the Part 1 corpus collection, inspection, cleaning, and splitting workflow."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = ("build_raw.py", "inspect_corpus.py", "clean_corpus.py", "split_corpus.py")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for step in STEPS:
        subprocess.run([sys.executable, str(ROOT / "src" / step)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
