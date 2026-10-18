"""Build the retrieval index (data/index/) from data/processed/*.jsonl.

Usage:  python src/build_index.py [--force]
Run src/parse_documents.py first.
"""
from __future__ import annotations

import argparse
import logging
import sys

from pldc.embeddings import GeminiEmbedder
from pldc.search import build_index


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = ap.parse_args()
    status = build_index(GeminiEmbedder(), force=args.force)
    print("Index already up to date (use --force to rebuild)." if status == "unchanged"
          else f"Index {status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
