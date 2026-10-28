"""
hard_cases.py - hand-written hard cases (data/hard_cases/) with inline answer keys.

Cases are written with inline markers, e.g. "Seen with [[NAME:Tan Ah Kow]] today."
parse() turns each line into plain text plus answer-key spans. Scoring is shared with
eval/score_deid.py:

  python -m eval.score_deid --split hard [--mode patterns|pipeline]
"""
import re

from common.config import ROOT

CASES_FILE = ROOT / "data" / "hard_cases" / "hard_cases.txt"
MARK = re.compile(r"\[\[([A-Z]+):(.+?)\]\]", re.S)


def parse(path=CASES_FILE) -> list[dict]:
    cases, category = [], "uncategorised"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            category = line[2:].strip()
            continue
        line = line.replace("\\n", "\n")
        text, spans, pos = "", [], 0
        for m in MARK.finditer(line):
            text += line[pos:m.start()]
            start = len(text)
            text += m.group(2)
            spans.append({"type": m.group(1), "text": m.group(2), "start": start, "end": len(text)})
            pos = m.end()
        text += line[pos:]
        cases.append({"category": category, "text": text, "phi": spans})
    return cases
