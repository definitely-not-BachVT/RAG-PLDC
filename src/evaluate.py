"""Evaluate answer accuracy on the question banks (see pldc/evaluation.py).

Examples:
  python src/evaluate.py --limit 60                     # quick stratified sample, all 3 modes
  python src/evaluate.py --modes none,hybrid --provider deepseek
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

from pldc.config import DATA_DIR, get_settings
from pldc.embeddings import GeminiEmbedder
from pldc.evaluation import MODES, row_dict, run_one, sample, summarize
from pldc.llm import DeepSeekProvider, GeminiProvider
from pldc.quiz import QuestionBank
from pldc.search import Retriever


def make_provider(name: str):
    s = get_settings()
    if name == "deepseek":
        if not s.deepseek_api_key:
            sys.exit("DEEPSEEK_API_KEY is not set")
        return DeepSeekProvider(s.deepseek_api_key, s.deepseek_model, s.deepseek_base_url, s.llm_timeout * 2)
    if not s.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set")
    return GeminiProvider(s.google_api_key, s.gemini_models[0], s.llm_timeout * 2)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size (default: all)")
    ap.add_argument("--modes", default=",".join(MODES), help="comma list of none,bm25,hybrid")
    ap.add_argument("--provider", choices=["deepseek", "gemini"], default="deepseek")
    ap.add_argument("--source", choices=["nganhang_tn", "eduquiz"], default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    if bad := set(modes) - set(MODES):
        sys.exit(f"unknown modes: {bad}")
    questions = sample(QuestionBank(), args.limit, args.seed, args.source)
    provider = make_provider(args.provider)
    retrievers = {}
    if "bm25" in modes:
        retrievers["bm25"] = Retriever(None)
    if "hybrid" in modes:
        retrievers["hybrid"] = Retriever(GeminiEmbedder())

    out_dir = DATA_DIR / "eval" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(q, m) for m in modes for q in questions]
    print(f"{len(questions)} questions x {len(modes)} modes = {len(jobs)} calls with {provider.name}"
          f" ({getattr(provider, 'model', '')}) -> {out_dir}")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool, (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        futures = [pool.submit(run_one, q, m, provider, retrievers) for q, m in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            f.write(json.dumps(row_dict(row), ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)")

    summ = summarize(rows)
    summ["config"] = {"provider": provider.name, "model": getattr(provider, "model", None),
                      "questions": len(questions), "modes": modes, "seed": args.seed,
                      "limit": args.limit, "source": args.source}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nmode     accuracy   n  errors unparsed avg_s")
    for mode in modes:
        m = summ["modes"].get(mode, {})
        acc = f"{m['accuracy']:.1%}" if m.get("accuracy") is not None else "-"
        print(f"{mode:<8} {acc:>8} {m.get('n', 0):>4} {m.get('errors', 0):>6} {m.get('unparsed', 0):>8} "
              f"{m.get('avg_seconds', 0):>5}")
        for g, v in m.get("groups", {}).items():
            if "chương" not in g:
                print(f"           {g}: {v['accuracy']:.1%} (n={v['n']})")
        if mode == "hybrid" and m.get("retrieval_used", {}).get("bm25"):
            print(f"           ! {m['retrieval_used']['bm25']} câu hybrid đã rơi về BM25 (embedding lỗi/hết hạn mức)")
        if "vs_none" in m:
            print(f"           so với 'none': giúp đúng {len(m['vs_none']['helped'])}, "
                  f"làm sai {len(m['vs_none']['hurt'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
