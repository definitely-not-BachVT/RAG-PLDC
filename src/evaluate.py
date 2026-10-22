"""Evaluate answer accuracy on the question banks (see pldc/evaluation.py).

Examples:
  python src/evaluate.py --limit 60                     # quick stratified sample, all 3 modes
  python src/evaluate.py --modes none,hybrid --model gemini-flash-lite-latest
  python src/evaluate.py                                # everything (676 questions x 3 modes)
Results go to data/eval/<timestamp>/ (rows.jsonl + summary.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pldc.config import DATA_DIR, get_settings, index_dir_for
from pldc.embeddings import GeminiEmbedder, LocalEmbedder
from pldc.evaluation import (MODES, VARIANTS, Row, ThrottledCachedEmbedder, row_dict, run_one, sample,
                             summarize)
from pldc.llm import GeminiProvider
from pldc.quiz import QuestionBank
from pldc.search import Retriever


def make_provider(model: str | None):
    s = get_settings()
    if not s.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set")
    return GeminiProvider(s.google_api_key, model or s.gemini_models[0], s.llm_timeout * 2)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size (default: all)")
    ap.add_argument("--modes", default="none,bm25,hybrid", help="comma list of: " + ",".join(MODES))
    ap.add_argument("--model", default=None, help="Gemini model (default: first of GEMINI_MODELS)")
    ap.add_argument("--source", choices=["nganhang_tn", "eduquiz"], default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="add/replace these modes in an existing run (rows of other modes are kept)")
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    if bad := set(modes) - set(MODES):
        sys.exit(f"unknown modes: {bad}")
    questions = sample(QuestionBank(), args.limit, args.seed, args.source)
    provider = make_provider(args.model)
    retrievers = {}
    kinds = {VARIANTS[m][0] for m in modes if m != "none"}
    if "bm25" in kinds:
        retrievers["bm25"] = Retriever(None)
    embedder = None
    if "hybrid" in kinds:
        # Free tier: ~100 embeddings/minute. Throttle + cache, or hybrid silently degrades to BM25.
        embedder = ThrottledCachedEmbedder(GeminiEmbedder(), DATA_DIR / "eval" / "query_vectors.npz")
        retrievers["hybrid"] = Retriever(embedder)
    if "local" in kinds:
        local = LocalEmbedder()
        retrievers["local"] = Retriever(local, index_dir=index_dir_for(local.model_id))

    out_dir = args.run_dir or DATA_DIR / "eval" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Row] = []
    if args.run_dir and (out_dir / "rows.jsonl").exists():
        for line in (out_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = Row(**json.loads(line))
            if r.mode not in modes:
                kept.append(r)
    jobs = [(q, m) for m in modes for q in questions]
    print(f"{len(questions)} questions x {len(modes)} modes = {len(jobs)} calls with {provider.name}"
          f" ({getattr(provider, 'model', '')}) -> {out_dir}", flush=True)

    rows: list[Row] = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, q, m, provider, retrievers) for q, m in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
                if embedder:
                    embedder.save()
    rows = kept + rows
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row_dict(row), ensure_ascii=False) + "\n")

    summ = summarize(rows)
    all_modes = [m for m in MODES if any(r.mode == m for r in rows)]
    summ["config"] = {"provider": provider.name, "model": getattr(provider, "model", None),
                      "questions": len(questions), "modes": all_modes, "seed": args.seed,
                      "limit": args.limit, "source": args.source}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nmode     accuracy   n  errors unparsed avg_s")
    for mode in all_modes:
        m = summ["modes"].get(mode, {})
        acc = f"{m['accuracy']:.1%}" if m.get("accuracy") is not None else "-"
        print(f"{mode:<8} {acc:>8} {m.get('n', 0):>4} {m.get('errors', 0):>6} {m.get('unparsed', 0):>8} "
              f"{m.get('avg_seconds', 0):>5}")
        for g, v in m.get("groups", {}).items():
            if "chương" not in g:
                print(f"           {g}: {v['accuracy']:.1%} (n={v['n']})")
        if mode not in ("none", "bm25") and m.get("retrieval_used", {}).get("bm25"):
            print(f"           ! {m['retrieval_used']['bm25']} câu hybrid đã rơi về BM25 (embedding lỗi/hết hạn mức)")
        if "vs_none" in m:
            print(f"           so với 'none': giúp đúng {len(m['vs_none']['helped'])}, "
                  f"làm sai {len(m['vs_none']['hurt'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
