"""
score_deid.py - score de-identification against the planted answer key (data/truth/phi).

Leak recall (headline): a truth span is caught only if predicted spans cover every
non-space character of it, whatever their type. A miss is a privacy leak.
Typed recall: covered by spans of the same type. Exact: same start and end.
Precision: predicted spans that overlap no truth span are false positives.

Writes runs/deid_<split>_<mode>.json and runs/deid_<split>_<mode>_misses.txt.

Usage:
  python -m eval.score_deid --split dev
  python -m eval.score_deid --split test --final     # final evaluation only
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict

from common.config import ROOT
from common.data import load_phi_truth, load_report, manifest
from deid.patterns import find_pattern_phi
from deid.types import PHI_TYPES

RUNS = ROOT / "runs"


def covered(span: dict, preds, text: str, same_type: bool = False) -> bool:
    chars = {i for p in preds if not same_type or p.type == span["type"] for i in range(p.start, p.end)}
    return all(i in chars for i in range(span["start"], span["end"]) if not text[i].isspace())


def context(text: str, start: int, end: int, width: int = 40) -> str:
    return (text[max(0, start - width):start] + "[[" + text[start:end] + "]]" + text[end:end + width]).replace("\n", " | ")


def score(split: str, mode: str) -> dict:
    stats = defaultdict(Counter)       # type -> truth, leak, typed, exact
    fp_by_source = Counter()
    tp_by_source = Counter()
    misses, false_pos = [], []

    rows = manifest(split)
    t0 = time.perf_counter()
    for row in rows:
        rid = row["report_id"]
        text = load_report(rid)
        preds = find_pattern_phi(text, report_id=rid)
        truth = load_phi_truth(rid)["phi"]

        for s in truth:
            st = stats[s["type"]]
            st["truth"] += 1
            if covered(s, preds, text):
                st["leak"] += 1
            else:
                misses.append(f"{rid}\t{s['type']}\t{context(text, s['start'], s['end'])}")
            st["typed"] += covered(s, preds, text, same_type=True)
            st["exact"] += any(p.start == s["start"] and p.end == s["end"] for p in preds)

        for p in preds:
            if any(p.start < s["end"] and s["start"] < p.end for s in truth):
                tp_by_source[p.source] += 1
            else:
                fp_by_source[p.source] += 1
                false_pos.append(f"{rid}\t{p.type}\t{p.source}\t{context(text, p.start, p.end)}")
    seconds = time.perf_counter() - t0

    total = sum((stats[t] for t in stats), Counter())
    n_pred = sum(tp_by_source.values()) + sum(fp_by_source.values())
    return {
        "split": split, "mode": mode, "reports": len(rows), "seconds": round(seconds, 2),
        "overall": {
            "truth": total["truth"],
            "leak_recall": round(total["leak"] / total["truth"], 4),
            "typed_recall": round(total["typed"] / total["truth"], 4),
            "exact_recall": round(total["exact"] / total["truth"], 4),
            "predicted": n_pred,
            "precision": round(sum(tp_by_source.values()) / n_pred, 4) if n_pred else None,
        },
        "by_type": {
            t: {"truth": st["truth"], "leak_recall": round(st["leak"] / st["truth"], 4),
                "typed_recall": round(st["typed"] / st["truth"], 4),
                "exact_recall": round(st["exact"] / st["truth"], 4), "missed": st["truth"] - st["leak"]}
            for t in PHI_TYPES if (st := stats.get(t))
        },
        "false_positives_by_source": dict(fp_by_source.most_common()),
        "_misses": misses,
        "_false_positives": false_pos,
    }


def print_report(result: dict) -> None:
    o = result["overall"]
    print(f"De-id scoring: split={result['split']} mode={result['mode']} "
          f"reports={result['reports']} ({result['seconds']} s)\n")
    print(f"{'type':10} {'truth':>5} {'leak':>7} {'typed':>7} {'exact':>7} {'missed':>6}")
    for t, r in result["by_type"].items():
        print(f"{t:10} {r['truth']:>5} {r['leak_recall']:>7.1%} {r['typed_recall']:>7.1%} "
              f"{r['exact_recall']:>7.1%} {r['missed']:>6}")
    print(f"{'ALL':10} {o['truth']:>5} {o['leak_recall']:>7.1%} {o['typed_recall']:>7.1%} "
          f"{o['exact_recall']:>7.1%} {o['truth'] - round(o['leak_recall'] * o['truth']):>6}")
    print(f"\nprecision {o['precision']:.1%} ({o['predicted']} predicted spans)")
    if result["false_positives_by_source"]:
        print("false positives by source:", result["false_positives_by_source"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Score de-identification against data/truth/phi.")
    ap.add_argument("--split", default="dev", choices=["dev", "extra", "test"])
    ap.add_argument("--mode", default="patterns", choices=["patterns"])
    ap.add_argument("--final", action="store_true", help="required to score the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is for the final evaluation only; add --final if this is it")

    result = score(args.split, args.mode)
    print_report(result)

    RUNS.mkdir(exist_ok=True)
    stem = RUNS / f"deid_{args.split}_{args.mode}"
    misses, false_pos = result.pop("_misses"), result.pop("_false_positives")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (stem.parent / f"{stem.name}_misses.txt").write_text(
        "# MISSES (leaks)\n" + "\n".join(misses) + "\n\n# FALSE POSITIVES\n" + "\n".join(false_pos) + "\n",
        encoding="utf-8")
    print(f"\nwrote {stem.name}.json and {stem.name}_misses.txt in runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
