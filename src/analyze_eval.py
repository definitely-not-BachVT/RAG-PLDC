"""Second pass over an evaluation run: answer-key audit and retrieval support.

  python src/analyze_eval.py data/eval/<run> keys      # suspect answer keys -> suspects.md/.csv
  python src/analyze_eval.py data/eval/<run> support   # retrieval support per mode -> support.json
Uses the retrieved chunk ids stored in the run, so no embedding calls are made.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pldc.chunking import Chunk
from pldc.config import INDEX_DIR, get_settings
from pldc.judging import audit_key, judge_support
from pldc.llm import DeepSeekProvider
from pldc.quiz import QuestionBank
from pldc.search import Hit

VERDICT_VI = {"KEY_WRONG": "Đáp án gốc có thể sai", "KEY_RIGHT": "Đáp án gốc đúng (model sai)",
              "UNCLEAR": "Chưa rõ, cần người duyệt"}


def load_run(run: Path) -> dict[str, dict[str, dict]]:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_q[r["qid"]][r["mode"]] = r
    return by_q


def load_chunks() -> dict[str, Chunk]:
    lines = (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return {c.id: c for c in (Chunk(**json.loads(l)) for l in lines)}


def hits_for(row: dict | None, chunks: dict[str, Chunk]) -> list[Hit]:
    return [Hit(chunks[i], 0.0) for i in (row or {}).get("chunk_ids", []) if i in chunks]


def provider(model: str) -> DeepSeekProvider:
    s = get_settings()
    return DeepSeekProvider(s.deepseek_api_key, model, s.deepseek_base_url, timeout=180)


def suspects(by_q: dict) -> list[str]:
    """Questions every mode got 'wrong' (and none errored)."""
    return sorted(q for q, modes in by_q.items()
                  if modes and all(not r["error"] and not r["correct"] for r in modes.values()))


def run_keys(run: Path, model: str, workers: int) -> None:
    bank, chunks, by_q = QuestionBank(), load_chunks(), load_run(run)
    ids = suspects(by_q)
    judge = provider(model)
    print(f"{len(ids)} suspect questions; judging with {model}")

    def work(qid):
        modes = by_q[qid]
        ctx = (hits_for(modes.get("hybrid"), chunks) or hits_for(modes.get("bm25"), chunks))
        return qid, ctx, audit_key(bank.by_id[qid], ctx, [r["predicted"] for r in modes.values()], judge)

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(work, ids))

    rows = []
    for qid, ctx, v in results:
        q, modes = bank.by_id[qid], by_q[qid]
        rows.append({
            "qid": qid, "bank": q.source, "chapter": q.chapter or "", "question": q.question,
            "options": " | ".join(f"{o['label']}. {o['text']}" for o in q.options),
            "key": ",".join(q.answers),
            "model_answers": "; ".join(f"{m}:{','.join(r['predicted']) or '-'}" for m, r in sorted(modes.items())),
            "judge_answer": ",".join(v.judge_answer), "confidence": v.confidence,
            "verdict": v.verdict, "reason": v.reason,
            "judge_sources": "; ".join(ctx[n - 1].chunk.citation for n in v.sources if 0 < n <= len(ctx)),
        })
    order = {"KEY_WRONG": 0, "UNCLEAR": 1, "KEY_RIGHT": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], r["qid"]))

    with (run / "suspects.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["qid"])
        w.writeheader()
        w.writerows(rows)
    md = [f"# Đáp án đáng ngờ ({len(rows)} câu)\n",
          f"Giám khảo: `{model}`, tự giải độc lập (không được xem đáp án gốc), có trích đoạn giáo trình.\n"]
    for verdict in order:
        group = [r for r in rows if r["verdict"] == verdict]
        md.append(f"\n## {VERDICT_VI[verdict]} ({len(group)})\n")
        for r in group:
            md.append(f"\n### `{r['qid']}` {r['question']}\n")
            md += [f"- {o}" for o in r["options"].split(" | ")]
            md.append(f"\n**Đáp án gốc:** {r['key']} · **model:** {r['model_answers']} · "
                      f"**giám khảo:** {r['judge_answer'] or '-'} ({r['confidence'] or '-'})  ")
            md.append(f"**Lý do:** {r['reason']}  ")
            if r["judge_sources"]:
                md.append(f"**Căn cứ:** {r['judge_sources']}")
    (run / "suspects.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    proposed = {r["qid"]: {"answers": r["judge_answer"].split(","),
                           "note": f"Proposed by {model} audit: {r['reason']}"}
                for r in rows if r["verdict"] == "KEY_WRONG" and r["confidence"] == "cao"}
    (run / "answer_overrides.proposed.json").write_text(
        json.dumps({"questions": proposed}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(Counter(r["verdict"] for r in rows), f"-> {run / 'suspects.md'}")
    print(f"{len(proposed)} high-confidence key fixes proposed -> {run / 'answer_overrides.proposed.json'}")


def run_support(run: Path, model: str, workers: int) -> None:
    bank, chunks, by_q = QuestionBank(), load_chunks(), load_run(run)
    excluded = set(suspects(by_q))  # don't measure retrieval against a key that may be wrong
    modes = [m for m in ("bm25", "hybrid") if any(m in v for v in by_q.values())]
    jobs = [(qid, m) for qid in sorted(by_q) if qid not in excluded for m in modes if m in by_q[qid]]
    judge = provider(model)
    print(f"{len(jobs)} judgements ({len(excluded)} suspect questions excluded) with {model}")

    def work(job):
        qid, mode = job
        return qid, mode, judge_support(bank.by_id[qid], hits_for(by_q[qid][mode], chunks), judge)

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(work, jobs))
    with (run / "support_rows.jsonl").open("w", encoding="utf-8") as f:
        for qid, mode, label in results:
            f.write(json.dumps({"qid": qid, "mode": mode, "support": label}) + "\n")

    summary: dict = {}
    for mode in modes:
        labels = [(qid, l) for qid, m, l in results if m == mode]
        per_bank: dict[str, Counter] = defaultdict(Counter)
        for qid, l in labels:
            per_bank[bank.by_id[qid].source][l] += 1
        total = Counter(l for _, l in labels)
        n = sum(total.values())
        # does good context go with correct answers? (accuracy when context is sufficient vs not)
        acc = defaultdict(list)
        for qid, l in labels:
            acc[l].append(by_q[qid][mode]["correct"])
        summary[mode] = {
            "n": n, "counts": dict(total),
            "rate": {k: round(v / n, 4) for k, v in total.items()},
            "per_bank": {b: dict(c) for b, c in per_bank.items()},
            "accuracy_by_support": {k: round(sum(v) / len(v), 4) for k, v in acc.items()},
        }
    (run / "support.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, s in summary.items():
        r = s["rate"]
        print(f"{mode:<7} đủ căn cứ {r.get('DU', 0):.1%}  một phần {r.get('MOT_PHAN', 0):.1%}  "
              f"không {r.get('KHONG', 0):.1%}  lỗi {r.get('ERROR', 0):.1%}  (n={s['n']})")
        print(f"        độ chính xác theo mức căn cứ: {s['accuracy_by_support']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("task", choices=["keys", "support"])
    ap.add_argument("--model", default=None, help="judge model (keys: deepseek-v4-pro, support: deepseek-flash)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.task == "keys":
        run_keys(args.run, args.model or "deepseek-v4-pro", args.workers)
    else:
        run_support(args.run, args.model or "deepseek-flash", args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
