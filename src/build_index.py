"""Build the retrieval index from data/processed/*.jsonl.

Usage:  python src/build_index.py [--embedder gemini|local] [--force]
  gemini -> data/index/          (API, free tier ~100 texts/min, ~1000/day)
  local  -> data/index-bge-m3/   (bge-m3 on CPU, no quota; needs models/bge-m3)
Default embedder comes from EMBEDDER in .env. Run src/parse_documents.py first.
"""
from __future__ import annotations

import argparse
import logging
import sys

from pldc.config import get_settings, index_dir_for
from pldc.embeddings import GeminiEmbedder, LocalEmbedder
from pldc.search import build_index


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", choices=["gemini", "local"], default=get_settings().embedder)
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = ap.parse_args()
    embedder = LocalEmbedder() if args.embedder == "local" else GeminiEmbedder()
    index_dir = index_dir_for(embedder.model_id)
    status = build_index(embedder, index_dir=index_dir, force=args.force)
    print(f"{index_dir}: " + ("already up to date (use --force to rebuild)." if status == "unchanged"
                              else f"{status}."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
