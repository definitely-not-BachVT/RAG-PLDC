"""Compare retrieval set-ups (chunking x first stage x reranker) on the question banks.

For a sample of questions, every set-up retrieves its top k passages for the question text. An LLM
judge labels each distinct passage once: does it give grounds for the correct answer? Labels are
cached in data/eval/retrieval_labels.jsonl (keyed by question, key and passage text), so re-runs and
new set-ups only pay for passages not seen before.

  python src/eval_retrieval.py --index v1=path/to/old/index --index v2=data/index-bge-m3 --limit 150
Each --index is a directory written by build_index.py; its manifest says which embedder it needs.
Per index the set-ups are: bm25, hybrid, hybrid+rerank (with --rerank).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pldc.config import DATA_DIR, get_settings
from pldc.embeddings import GeminiEmbedder, LocalEmbedder
from pldc.evaluation import ThrottledCachedEmbedder, question_text, sample
from pldc.judging import label_relevance
from pldc.llm import GeminiProvider
from pldc.quiz import Question, QuestionBank
from pldc.rerank import Reranker
from pldc.search import Retriever

LABELS = DATA_DIR / "eval" / "retrieval_labels.jsonl"
JUDGE_ATTEMPTS = 4


def label_key(q: Question, body: str) -> str:
    return hashlib.sha256(f"{q.id}|{','.join(sorted(q.answers))}|{body}".encode()).hexdigest()[:24]


def build_setups(indexes: list[tuple[str, Path]], rerank: bool, candidates: int) -> dict[str, Retriever]:
    reranker = Reranker() if rerank else None
    if reranker and not reranker.on_gpu:
        print("! reranker on the CPU: expect ~25 s per question", flush=True)
    embedders: dict[str, object] = {}
    setups = {}
    for label, path in indexes:
        model = json.loads((path / "manifest.json").read_text(encoding="utf-8"))["embed_model"]
        if model not in embedders:
            if model.startswith("gemini:"):
                embedders[model] = ThrottledCachedEmbedder(GeminiEmbedder(),
                                                           DATA_DIR / "eval" / "query_vectors.npz")
            else:
                embedders[model] = LocalEmbedder()
        emb = embedders[model]
        setups[f"{label}/bm25"] = Retriever(None, index_dir=path)
        setups[f"{label}/hybrid"] = Retriever(emb, index_dir=path)
        if reranker:
            setups[f"{label}/hybrid+rerank"] = Retriever(emb, index_dir=path, reranker=reranker,
                                                         rerank_candidates=candidates)
    return setups


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--candidates", type=int, default=get_settings().rerank_candidates)
    ap.add_argument("--judge", default=None, help="Gemini model (default: first of GEMINI_MODELS)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    ap.add_argument("--rows", type=Path, default=None,
                    help="write per-question labels (JSONL) here, for paired comparisons")
    args = ap.parse_args()

    indexes = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in args.index]
    questions = sample(QuestionBank(), args.limit, args.seed)
    setups = build_setups(indexes, args.rerank, args.candidates)
    print(f"{len(questions)} questions x {len(setups)} set-ups, k={args.k}", flush=True)

    # 1. Retrieve (sequential per set-up: the local models run one batch at a time anyway).
    results: dict[str, dict[str, list]] = {}
    for name, r in setups.items():
        t0 = time.perf_counter()
        results[name] = {q.id: r.search_with_mode(question_text(q), k=args.k)[0] for q in questions}
        print(f"  {name:28} {1000 * (time.perf_counter() - t0) / len(questions):7.0f} ms/question",
              flush=True)
    for emb in {id(r.embedder): r.embedder for r in setups.values()}.values():
        if isinstance(emb, ThrottledCachedEmbedder):
            emb.save()

    # 2. Label every distinct passage that any set-up returned.
    labels: dict[str, bool] = {}
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels[row["key"]] = row["relevant"]
    s = get_settings()
    judge = GeminiProvider(s.google_api_key, args.judge or s.gemini_models[0], s.llm_timeout * 2)
    todo: dict[str, list[str]] = {}
    for q in questions:
        bodies = {h.chunk.body for res in results.values() for h in res[q.id]}
        todo[q.id] = sorted(b for b in bodies if label_key(q, b) not in labels)
    pending = [q for q in questions if todo[q.id]]
    print(f"judge {judge.model}: {sum(map(len, todo.values()))} new passages "
          f"in {len(pending)} questions", flush=True)
    lock = threading.Lock()

    def work(q: Question) -> None:
        for attempt in range(JUDGE_ATTEMPTS):
            try:
                rel = label_relevance(q, todo[q.id], judge)
                break
            except Exception as exc:  # noqa: BLE001 - rate limits / malformed replies: retry
                if attempt == JUDGE_ATTEMPTS - 1:
                    print(f"  ! {q.id}: {exc}", flush=True)
                    return
                time.sleep(10 * 2 ** attempt)
        with lock:
            with LABELS.open("a", encoding="utf-8") as f:
                for body, ok in zip(todo[q.id], rel):
                    labels[label_key(q, body)] = ok
                    f.write(json.dumps({"qid": q.id, "key": label_key(q, body), "relevant": ok}) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, pending))

    # 3. Metrics. Questions with a passage left unlabeled (judge failed) are skipped.
    if args.rows:
        with args.rows.open("w", encoding="utf-8") as f:
            for name, res in results.items():
                for q in questions:
                    keys = [label_key(q, h.chunk.body) for h in res[q.id]]
                    if all(k in labels for k in keys):
                        f.write(json.dumps({"setup": name, "qid": q.id,
                                            "relevant": [labels[k] for k in keys]}) + "\n")
    summary = {}
    for name, res in results.items():
        hit, hit3, mrr, prec, chars, n = 0, 0, 0.0, 0.0, 0, 0
        for q in questions:
            keys = [label_key(q, h.chunk.body) for h in res[q.id]]
            if any(k not in labels for k in keys):
                continue
            rel = [labels[k] for k in keys]
            n += 1
            hit += any(rel)
            hit3 += any(rel[:3])
            mrr += next((1 / r for r, ok in enumerate(rel, start=1) if ok), 0.0)
            prec += sum(rel) / max(len(rel), 1)
            chars += sum(len(h.chunk.body) for h in res[q.id])
        summary[name] = {"n": n, f"hit@{args.k}": round(hit / n, 4), "hit@3": round(hit3 / n, 4),
                         "mrr": round(mrr / n, 4), f"precision@{args.k}": round(prec / n, 4),
                         "context_chars": round(chars / n)} if n else {"n": 0}
    print(f"\n{'set-up':28} {'n':>4} {'hit@' + str(args.k):>7} {'hit@3':>7} {'MRR':>6} "
          f"{'prec':>6} {'chars':>6}")
    for name, m in summary.items():
        if m["n"]:
            print(f"{name:28} {m['n']:4} {m[f'hit@{args.k}']:7.1%} {m['hit@3']:7.1%} {m['mrr']:6.3f} "
                  f"{m[f'precision@{args.k}']:6.1%} {m['context_chars']:6}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"config": vars(args) | {"index": args.index, "out": str(args.out)},
                                        "setups": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
