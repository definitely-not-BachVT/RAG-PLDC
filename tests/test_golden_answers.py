"""End-to-end answer quality on golden questions, with the real LLM chain (Gemini models in turn).

Run:  pytest -m live tests/test_golden_answers.py      (costs a few cents, ~2-4 minutes)
Each answer is also logged to data/eval/golden-<date>.jsonl for inspection.

Checked per case kind:
  grounded     key facts present, citations valid, the expected textbook section is *cited*,
               answer judged faithful to its sources
  out_of_scope politely declined (no recipe!)
  injection    the system prompt is not revealed
  currency     mentions the current law (giáo trình 2019 is outdated here)
and for every case: time to first token and total time within budget.
"""
import json
import os
import time
from datetime import date

import pytest

from golden import CHAT_CASES, EXPLAIN_CASES
from pldc.answer_checks import (cited_numbers, cited_sections, invalid_citations, judge_faithfulness,
                                missing_facts, section_matches, stated_correct_letter)
from pldc.chat import answer
from pldc.config import DATA_DIR, get_settings
from pldc.llm import GeminiProvider, build_llm
from pldc.quiz import QuestionBank, explain
from pldc.search import build_retriever

pytestmark = pytest.mark.live

MAX_FIRST_TOKEN_S = float(os.environ.get("GOLDEN_MAX_FIRST_TOKEN_S", 25))
MAX_TOTAL_S = float(os.environ.get("GOLDEN_MAX_TOTAL_S", 90))
LOG = DATA_DIR / "eval" / f"golden-{date.today():%Y%m%d}.jsonl"


@pytest.fixture(scope="module")
def stack():
    s = get_settings()
    if not s.google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    judge = GeminiProvider(s.google_api_key, s.gemini_models[0], 60)
    return build_llm(), build_retriever(), judge


def run_stream(stream_result) -> str:
    return "".join(stream_result.chunks)


def log(record: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.mark.parametrize("case", CHAT_CASES, ids=[c["id"] for c in CHAT_CASES])
def test_chat_answer(case, stack):
    llm, retriever, judge = stack
    t0 = time.perf_counter()
    # answer() returns once the first token has arrived (fallback only happens before it), so
    # this is retrieval + time to first token: what the learner waits before text appears.
    res = answer(case["question"], case.get("history", []), retriever, llm)
    first_s = time.perf_counter() - t0
    text = run_stream(res.stream)
    total_s = time.perf_counter() - t0
    hits = res.hits
    record = {"id": case["id"], "kind": case["kind"], "provider": res.stream.provider,
              "retrieval": retriever.last_mode, "first_token_s": round(first_s, 2),
              "total_s": round(total_s, 2), "cited": cited_numbers(text),
              "cited_sections": sorted(cited_sections(text, hits)), "answer": text}
    problems = []

    if case.get("must_include") and (missing := missing_facts(text, case["must_include"])):
        problems.append(f"missing facts {missing}")
    if leaked := [m for m in case.get("must_not", []) if m.lower() in text.lower()]:
        problems.append(f"must not contain {leaked}")

    if case["kind"] == "grounded":
        if not cited_numbers(text):
            problems.append("no citation")
        if bad := invalid_citations(text, len(hits)):
            problems.append(f"cites non-existent sources {bad} (only {len(hits)})")
        if not section_matches(cited_sections(text, hits), case["expected_sections"]):
            problems.append(f"expected section {case['expected_sections']} not cited "
                            f"(cited {sorted(cited_sections(text, hits))})")
        if judge is not None:
            verdict = judge_faithfulness(judge, text, hits)
            record["faithfulness"] = verdict
            if verdict == "KHONG":
                problems.append("judge: answer not supported by its sources")

    if first_s > MAX_FIRST_TOKEN_S:
        problems.append(f"first token after {first_s:.1f}s > {MAX_FIRST_TOKEN_S}s")
    if total_s > MAX_TOTAL_S:
        problems.append(f"answer took {total_s:.1f}s > {MAX_TOTAL_S}s")

    record["problems"] = problems
    log(record)
    assert not problems, f"{case['id']} ({res.stream.provider}): " + "; ".join(problems) \
        + f"\n--- answer ---\n{text[:1500]}"


@pytest.mark.parametrize("case", EXPLAIN_CASES, ids=[c["qid"] for c in EXPLAIN_CASES])
def test_quiz_explanation(case, stack):
    llm, retriever, _ = stack
    q = QuestionBank().by_id[case["qid"]]
    t0 = time.perf_counter()
    hits, stream = explain(q, case["chosen"], retriever, llm)
    text = run_stream(stream)
    total_s = time.perf_counter() - t0
    stated = stated_correct_letter(text)
    problems = []
    if stated is None:
        problems.append("does not state which option is correct")
    elif stated not in q.answers:
        problems.append(f"presents {stated} as correct, key is {q.answers}")
    if missing := missing_facts(text, case.get("must_include", [])):
        problems.append(f"missing facts {missing}")
    if bad := invalid_citations(text, len(hits)):
        problems.append(f"cites non-existent sources {bad}")
    if wrong_choice := [c for c in case["chosen"] if c not in q.answers]:
        if not any(f"{c}" in text for c in wrong_choice):
            problems.append(f"does not address the learner's wrong choice {wrong_choice}")
    if total_s > MAX_TOTAL_S:
        problems.append(f"explanation took {total_s:.1f}s > {MAX_TOTAL_S}s")
    log({"id": f"explain:{case['qid']}", "provider": stream.provider, "total_s": round(total_s, 2),
         "stated": stated, "key": q.answers, "problems": problems, "answer": text})
    assert not problems, f"{case['qid']} ({stream.provider}): " + "; ".join(problems) \
        + f"\n--- explanation ---\n{text[:1500]}"
