"""Golden cases shared by the answer-quality tests (data/golden/*.jsonl)."""
import json

from pldc.config import DATA_DIR

GOLDEN_DIR = DATA_DIR / "golden"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLDEN_DIR / name).read_text(encoding="utf-8").splitlines() if line]


CHAT_CASES = load("chat_cases.jsonl")
EXPLAIN_CASES = load("explain_cases.jsonl")
GROUNDED = [c for c in CHAT_CASES if c["kind"] == "grounded"]


def retrieval_query(case: dict) -> str:
    """Same rule as pldc.chat.retrieval_query: a short follow-up is searched with the previous question."""
    from pldc.chat import retrieval_query as rq
    return rq(case["question"], case.get("history", []))
