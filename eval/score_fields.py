"""
score_fields.py - extraction scores from the "test then learn" log (runs/score_log.jsonl).

Each log line holds PrivLab's prediction (saved before feedback) and the user's confirmed answer.
Per field: correct = TP; wrong counts as one FP and one FN; extra = FP; missed = FN; true_empty is
correct but has no value. Also the learning curve (running accuracy, report by report) and the
number of confirmations until the field first became Ready.

Writes runs/fields_<phase>[_<split>]_<YYYYMMDD-HHMMSS>.json (only the latest of each kind is kept).

Usage:
  python -m eval.score_fields --phase teach
  python -m eval.score_fields --phase test --split test --final
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from common.config import ROOT
from extract.learn import read_log

RUNS = ROOT / "runs"


def score(entries: list[dict]) -> dict:
    by_field = defaultdict(list)
    for e in entries:
        by_field[e["field"]].append(e)
    out = {}
    for key, rows in by_field.items():
        n = {k: sum(r["outcome"] == k for r in rows) for k in ("correct", "wrong", "missed", "extra", "true_empty")}
        tp, fp, fn = n["correct"], n["wrong"] + n["extra"], n["wrong"] + n["missed"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if tp + fp + fn else None
        curve, ok = [], 0
        for i, r in enumerate(rows, 1):
            ok += r["outcome"] in ("correct", "true_empty")
            curve.append(round(ok / i, 3))
        to_ready = next((i for i, r in enumerate(rows, 1) if r.get("readiness_after") == "ready"), None)
        out[key] = {"reports": len(rows), **n,
                    "precision": None if precision is None else round(precision, 3),
                    "recall": None if recall is None else round(recall, 3),
                    "f1": None if f1 is None else round(f1, 3),
                    "learning_curve": curve, "confirmations_to_ready": to_ready}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Score extraction from the test-then-learn log.")
    ap.add_argument("--phase", default="teach", choices=["teach", "test"])
    ap.add_argument("--split", default=None, choices=["dev", "extra", "test"])
    ap.add_argument("--final", action="store_true", help="required to score the test split")
    ap.add_argument("--log", default=None, help="another score log, e.g. runs/try_score_log.jsonl")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is for the final evaluation only; add --final if this is it")

    log = (ROOT / args.log) if args.log else None
    entries = [e for e in read_log(log) if e["phase"] == args.phase and (args.split is None or e["split"] == args.split)]
    if not entries:
        print(f"no matching confirmations in {args.log or 'runs/score_log.jsonl'} yet")
        return 1
    result = score(entries)
    print(f"{'field':16} {'n':>3} {'P':>6} {'R':>6} {'F1':>6} {'ready after':>12}  learning curve")
    for key, s in result.items():
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{key:16} {s['reports']:>3} {fmt(s['precision']):>6} {fmt(s['recall']):>6} {fmt(s['f1']):>6} "
              f"{s['confirmations_to_ready'] or '-':>12}  {s['learning_curve']}")

    now = datetime.now(timezone(timedelta(hours=8)))
    kind = f"fields_{args.phase}{'_' + args.split if args.split else ''}{'_try' if args.log else ''}"
    RUNS.mkdir(exist_ok=True)
    for old in RUNS.glob(f"{kind}_2*.json"):
        old.unlink()
    path = RUNS / f"{kind}_{now:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"run_at": now.isoformat(timespec="seconds"), "phase": args.phase,
                                "split": args.split, "fields": result}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.name} in runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
