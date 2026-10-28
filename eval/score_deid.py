"""
score_deid.py - score de-identification against an answer key, layer by layer.

Modes:
  patterns  Presidio patterns only (Layer 1 baseline)
  pipeline  the full review-first flow without Muse Spark (no API calls)
  spark     the full flow with the Muse Spark check: live calls, needs --allow-api

The simulated user confirms without changes and hides everything flagged; learning is on,
reports are processed in order with one fresh profile. Types the profile keeps on purpose
(clinical dates by default) are reported as "kept by choice", not as leaks.

Headline numbers:
  leak recall   share of identifiers fully covered in the final output (a miss is a leak)
  reached API   share of identifiers still unmasked when Muse Spark is called (Layer 1 misses)
  repeat misses Layer 1 missing something it had already learned (target 0)

Usage:
  python -m eval.score_deid --split dev [--mode pipeline] [--show-misses]
  python -m eval.score_deid --split hard
  python -m eval.score_deid --split test --final      # final evaluation only
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict

from common.config import ROOT
from common.data import load_phi_truth, load_report, manifest
from deid import pipeline
from deid.patterns import find_pattern_phi
from deid.profile import new_profile, removed_types
from deid.surrogates import norm
from deid.types import PHI_TYPES

RUNS = ROOT / "runs"


def covered(span: dict, preds, text: str, same_type: bool = False) -> bool:
    chars = {i for p in preds if not same_type or p.type == span["type"] for i in range(p.start, p.end)}
    return all(i in chars for i in range(span["start"], span["end"]) if not text[i].isspace())


def context(text: str, start: int, end: int, width: int = 40) -> str:
    return (text[max(0, start - width):start] + "[[" + text[start:end] + "]]" + text[end:end + width]).replace("\n", " ")


def load_cases(split: str) -> list[dict]:
    if split in ("hard", "redteam"):
        from eval.hard_cases import CASES_FILE, parse
        path = CASES_FILE if split == "hard" else CASES_FILE.with_name("redteam.txt")
        return [{"id": f"{split[0].upper()}{i:02d}", **c} for i, c in enumerate(parse(path), 1)]
    return [{"id": r["report_id"], "category": r["cancer_type"], "text": load_report(r["report_id"]),
             "phi": load_phi_truth(r["report_id"])["phi"]} for r in manifest(split)]


def predict(case: dict, mode: str, profile) -> dict:
    if mode == "patterns":
        spans = [s for s in find_pattern_phi(case["text"], case["id"]) if s.type in removed_types(profile)]
        return {"layer1": spans, "final": spans, "layer2": "skipped", "holds": 0}
    use_spark = mode == "spark"
    r = pipeline.deidentify_report(text=case["text"], report_id=case["id"], profile=profile)
    done = pipeline.confirm_deid(r["deid_id"], use_spark=use_spark)
    holds = len(done.get("hold", []))
    if done["next"] == "hold":                                     # simulated user: hide all
        done = pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=use_spark)
    if done["next"] == "flags":
        done = pipeline.confirm_deid_flags(r["deid_id"])
    run = pipeline._RUNS.pop(r["deid_id"])
    return {"layer1": run["layer1_final"], "final": run["final_spans"], "layer2": run["layer2"], "holds": holds}


def score(cases: list[dict], mode: str) -> dict:
    profile = new_profile()[0]
    removed = removed_types(profile)
    stats = defaultdict(Counter)                    # type -> truth, leak, typed, exact, reached_api
    by_category = defaultdict(Counter)
    fp_by_source, kept_by_choice, layer2 = Counter(), Counter(), Counter()
    repeat_misses, misses, false_pos, slowest, hold_total = 0, [], [], 0.0, 0

    for case in cases:
        text = case["text"]
        learned = {norm(r["text"]) for r in profile.always_remove}
        t0 = time.perf_counter()
        out = predict(case, mode, profile)
        slowest = max(slowest, time.perf_counter() - t0)
        layer2[out["layer2"]] += 1
        hold_total += out["holds"]
        truth = [s for s in case["phi"] if s["type"] in removed]
        kept_by_choice.update(s["type"] for s in case["phi"] if s["type"] not in removed)

        for s in truth:
            st, cat = stats[s["type"]], by_category[case["category"]]
            st["truth"] += 1
            cat["truth"] += 1
            if covered(s, out["final"], text):
                st["leak"] += 1
                cat["caught"] += 1
            else:
                misses.append(f"{case['id']}\t{s['type']}\t{context(text, s['start'], s['end'])}")
            st["typed"] += covered(s, out["final"], text, same_type=True)
            st["exact"] += any(p.start == s["start"] and p.end == s["end"] for p in out["final"])
            if not covered(s, out["layer1"], text):
                st["reached_api"] += 1
                repeat_misses += norm(s["text"]) in learned
        for p in out["final"]:
            if not any(p.start < s["end"] and s["start"] < p.end for s in case["phi"]):
                fp_by_source[p.source] += 1
                by_category[case["category"]]["false_alarms"] += 1
                false_pos.append(f"{case['id']}\t{p.type}\t{p.source}\t{context(text, p.start, p.end)}")

    total = sum(stats.values(), Counter())
    pct =lambda k: round(total[k] / total["truth"], 4) if total["truth"] else None
    return {
        "mode": mode, "cases": len(cases), "slowest_ms": round(slowest * 1000, 1),
        "overall": {"truth": total["truth"], "leak_recall": pct("leak"), "typed_recall": pct("typed"),
                    "exact_recall": pct("exact"), "reached_api": pct("reached_api"),
                    "remaining": round(1 - pct("leak"), 4) if total["truth"] else None,
                    "false_alarms": len(false_pos), "repeat_misses": repeat_misses,
                    "hold_flags_per_case": round(hold_total / len(cases), 2) if cases else 0},
        "by_type": {t: {"truth": st["truth"], "leak_recall": round(st["leak"] / st["truth"], 4),
                        "reached_api": st["reached_api"], "missed": st["truth"] - st["leak"]}
                    for t in PHI_TYPES if (st := stats.get(t))},
        "by_category": {c: dict(v) for c, v in by_category.items()},
        "kept_by_choice": dict(kept_by_choice), "layer2": dict(layer2),
        "false_positives_by_source": dict(fp_by_source.most_common()),
        "learned_entries": len(profile.always_remove),
        "_misses": misses, "_false_positives": false_pos,
    }


def print_report(r: dict, split: str) -> None:
    o = r["overall"]
    print(f"De-id scoring: split={split} mode={r['mode']} cases={r['cases']} (slowest {r['slowest_ms']} ms)\n")
    print(f"{'type':10} {'truth':>5} {'leak':>7} {'reached API':>12} {'missed':>6}")
    for t, s in r["by_type"].items():
        print(f"{t:10} {s['truth']:>5} {s['leak_recall']:>7.1%} {s['reached_api']:>12} {s['missed']:>6}")
    print(f"{'ALL':10} {o['truth']:>5} {o['leak_recall']:>7.1%}")
    print(f"\nreached the API (Layer 1 misses): {o['reached_api']:.1%}   remaining after all layers: {o['remaining']:.1%}")
    print(f"false alarms: {o['false_alarms']}   repeat misses: {o['repeat_misses']}   "
          f"hold flags/case: {o['hold_flags_per_case']}   "
          f"learned entries: {r['learned_entries']}   Muse Spark: {r['layer2']}")
    if r["kept_by_choice"]:
        print(f"kept by choice (not scored): {r['kept_by_choice']}")
    if split in ("hard", "redteam"):
        print(f"\n{'category':58} {'ids':>4} {'caught':>7} {'false alarms':>13}")
        for c, v in r["by_category"].items():
            print(f"{c:58} {v.get('truth', 0):>4} {v.get('caught', 0):>7} {v.get('false_alarms', 0):>13}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score de-identification against the answer key.")
    ap.add_argument("--split", default="dev", choices=["dev", "extra", "test", "hard", "redteam"])
    ap.add_argument("--mode", default="pipeline", choices=["patterns", "pipeline", "spark"])
    ap.add_argument("--final", action="store_true", help="required to score the test split")
    ap.add_argument("--allow-api", action="store_true", help="required for --mode spark (live calls)")
    ap.add_argument("--show-misses", action="store_true", help="write misses with context (dev/extra/hard only)")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is for the final evaluation only; add --final if this is it")
    if args.mode == "spark" and not args.allow_api:
        ap.error("--mode spark makes live Muse Spark calls; add --allow-api once approved")
    if args.show_misses and args.split == "test":
        ap.error("miss lists are dev/extra/hard only")

    result = score(load_cases(args.split), args.mode)
    print_report(result, args.split)
    RUNS.mkdir(exist_ok=True)
    stem = RUNS / f"deid_{args.split}_{args.mode}"
    misses, false_pos = result.pop("_misses"), result.pop("_false_positives")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.show_misses:
        (stem.parent / f"{stem.name}_misses.txt").write_text(
            "# MISSES (leaks)\n" + "\n".join(misses) + "\n\n# FALSE POSITIVES\n" + "\n".join(false_pos) + "\n",
            encoding="utf-8")
    print(f"\nwrote {stem.name}.json in runs/" + (" (+ misses file)" if args.show_misses else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
