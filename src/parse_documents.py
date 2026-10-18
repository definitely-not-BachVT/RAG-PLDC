"""Parse every PDF in data/ into JSONL files under data/processed/.

Usage:  python src/parse_documents.py [--data-dir data] [--out-dir data/processed]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from parsers import mcq_color, mcq_eduquiz, slides, textbook
from parsers.common import write_jsonl

# glob pattern -> (parser module, output file)
SOURCES = [
    ("giaotrinh_pldc_baigiang*.pdf", textbook, "textbook_sections.jsonl"),
    ("giaotrinh_pldc_slide*.pdf", slides, "slides.jsonl"),
    ("cauhoi_tno_abcd*.pdf", mcq_color, "questions_nganhang_tn.jsonl"),
    ("*PTIT*.pdf", mcq_eduquiz, "questions_eduquiz.jsonl"),
]
# Strings that must never survive parsing (template headers, page footers).
LEAK_MARKERS = ["HỌC THUYẾT KINH TẾ", "NGUYỄN VĂN A", "eduquiz.vn", "31/5/25",
                "TRƯỞNG BỘ MÔN", "NGƯỜI BIÊN SOẠN"]


def _find(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        sys.exit(f"Expected exactly one file for {pattern!r} in {data_dir}, found {len(matches)}")
    return matches[0]


def _report(name: str, records: list) -> list[str]:
    problems = []
    print(f"\n== {name}: {len(records)} records")
    blob = "\n".join(str(r) for r in records)
    for marker in LEAK_MARKERS:
        if marker.lower() in blob.lower():
            problems.append(f"{name}: leaked boilerplate {marker!r}")
    if not records:
        return problems + [f"{name}: no records"]
    if hasattr(records[0], "options"):
        print("  options per question:", dict(Counter(len(r.options) for r in records)))
        print("  answers per question:", dict(Counter(len(r.answers) for r in records)))
        print("  by chapter:", dict(sorted(Counter(r.chapter for r in records).items(), key=str)))
        review = [r for r in records if "needs_review" in r.flags]
        print(f"  needs_review: {len(review)}")
        for r in review:
            print(f"    {r.id} (p.{r.page}): {', '.join(f for f in r.flags if f != 'needs_review')}")
        forced = [r.id for r in records if "forced_split" in r.flags]
        if forced:
            print(f"  forced_split (option boundary guessed): {len(forced)}: {', '.join(forced)}")
    elif hasattr(records[0], "section_id"):
        print("  by kind:", dict(Counter(r.kind for r in records)))
        print("  by chapter:", dict(sorted(Counter(r.chapter for r in records).items())))
        lens = sorted(len(r.text) for r in records)
        print(f"  text chars: total {sum(lens)}, median {lens[len(lens) // 2]}, max {lens[-1]}")
    else:
        lens = sorted(len(r.text) for r in records)
        print(f"  text chars: total {sum(lens)}, median {lens[len(lens) // 2]}, max {lens[-1]}")
    return problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=root / "data")
    ap.add_argument("--out-dir", type=Path, default=root / "data" / "processed")
    args = ap.parse_args()

    problems = []
    for pattern, parser, out_name in SOURCES:
        pdf = _find(args.data_dir, pattern)
        records = parser.parse(pdf)
        write_jsonl(args.out_dir / out_name, records)
        problems += _report(f"{pdf.name} -> {out_name}", records)

    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print(f"\nOK, written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
