"""
qualify_model.py - qualify a model for the de-id name check before it is trusted.

Runs the frozen qualification set (data/hard_cases/qualify.txt: identifiers the local rules miss,
plus clinical words that must not be flagged) through the local layers and then the given model,
and records the result for the exact settings (provider, URL, model, effort, prompt version).
deid/pipeline.py only uses a qualified model; anything else falls back or warns.
The local name list is turned off here, so the model is measured on names the list would hold.

  catch         share of identifiers the local layers missed that the model put in findings
  false alarms  model findings on the negative lines
  pass          catch >= 90% and false alarms <= 1   (deid/qualify.py)

Live calls: one per line of the set (about 22). Needs --allow-api.

Usage:
  python -m eval.qualify_model --provider baseten --allow-api
  python -m eval.qualify_model --provider meta --allow-api
"""
import argparse
import sys

from common.config import ROOT
from deid import hold, pipeline, qualify
from deid.hold import suspects
from deid.mask import mask
from deid.names import find_names
from deid.profile import kept_types, new_profile
from eval.hard_cases import parse

QUALIFY_FILE = ROOT / "data" / "hard_cases" / "qualify.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Qualify a model for the de-id name check (live calls).")
    ap.add_argument("--provider", required=True, choices=["meta", "baseten"])
    ap.add_argument("--allow-api", action="store_true", help="required: makes one live call per line")
    args = ap.parse_args()
    if not args.allow_api:
        ap.error("this makes live calls (one per line of the qualification set); add --allow-api once approved")

    hold.USE_NAME_LIST = False
    cases = parse(QUALIFY_FILE)
    to_catch = caught = false_alarms = local = errors = 0
    misses, alarms = [], []
    for case in cases:
        profile = new_profile()[0]
        r = pipeline.deidentify_report(text=case["text"], report_id="qualify", profile=profile)
        done = pipeline.confirm_deid(r["deid_id"], use_spark=False)
        if done["next"] == "hold":
            pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=False)
        run = pipeline._RUNS.pop(r["deid_id"])
        spans, text = run["layer1_final"], case["text"]
        m = mask("qualify", text, spans, kept_types=kept_types(profile))
        try:
            found, _, _ = find_names(m, text, "qualify", provider=args.provider,
                                     suspects=suspects(m.text, (), m.tags))
        except Exception as e:                     # count as a miss for every identifier in the line
            errors += 1
            found = []
            print(f"  ! call failed ({type(e).__name__}) on: {text[:50]!r}")
        for s in case["phi"]:
            if any(x.start <= s["start"] and s["end"] <= x.end for x in spans):
                local += 1                         # caught locally: not the model's job here
                continue
            to_catch += 1
            if any(x.start <= s["start"] and s["end"] <= x.end for x in found):
                caught += 1
            else:
                misses.append(f"{s['text']!r} in {text!r}")
        if not case["phi"]:
            for x in found:
                false_alarms += 1
                alarms.append(f"{x.text!r} in {text!r}")

    catch = caught / to_catch if to_catch else 1.0
    result = qualify.record_result(args.provider, catch, false_alarms,
                                   {"identifiers": to_catch, "caught": caught, "caught_locally": local,
                                    "call_errors": errors, "lines": len(cases)})
    print(f"\n{args.provider} ({result['model']}, effort {result['effort']}): caught {caught}/{to_catch} "
          f"= {catch:.0%} (need {qualify.MIN_CATCH:.0%}), false alarms {false_alarms} "
          f"(max {qualify.MAX_FALSE_ALARMS}) -> {'QUALIFIED' if result['passed'] else 'NOT QUALIFIED'}")
    if local:
        print(f"note: {local} identifiers were already caught locally; the set should only hold ones the rules miss")
    for x in misses:
        print("  missed:", x)
    for x in alarms:
        print("  false alarm:", x)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
