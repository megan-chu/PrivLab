"""
try.py - try the review-first de-identification on one report in the terminal.

Shows what Layer 1 found (local only), lets you correct it, answers the hold rule, then
prints the tagged text. Uses a fresh in-memory profile, so nothing you do here is saved.
Muse Spark is NOT called unless you add --spark (one live call per report).

Usage:
  python -m deid.try PP0144            # dev or extra report
  python -m deid.try PP0144 --spark    # also run the Muse Spark check (live call)

Commands at the review prompt:
  unmark p3 p5          those detections are false alarms
  add NAME Tan Ah Kow   mark missed text (types: NAME, DOCTOR, PHONE, ADDRESS, ID, ...)
  Enter                 confirm
"""
import argparse
import sys

from common.data import load_phi_truth, load_report
from deid import pipeline
from deid.profile import new_profile
from deid.types import PHI_TYPES


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nstopped")


def show(items: list[dict]) -> None:
    for d in items:
        print(f"  {d['id']:4} {d['type']:9} {d.get('text', '')!r:32} {d['context']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Try de-identification on one report.")
    ap.add_argument("report_id")
    ap.add_argument("--spark", action="store_true", help="run the Muse Spark check (live API call)")
    args = ap.parse_args()

    if load_phi_truth(args.report_id)["split"] == "test":
        print(f"{args.report_id} is in the test split; open it only for the final evaluation.")
        return 2

    r = pipeline.deidentify_report(text=load_report(args.report_id), report_id=args.report_id,
                                   profile=new_profile()[0], show_text=True)
    print(f"\nLayer 1 (on this laptop, nothing sent): {len(r['detections'])} found {r['counts']}")
    print(f"kept by choice: {r['kept_by_choice']}\n")
    show(r["detections"])

    unmark, add = [], []
    while (cmd := ask("\nunmark <ids> | add <TYPE> <text> | Enter to confirm > ")):
        word, _, rest = cmd.partition(" ")
        if word == "unmark":
            unmark += rest.split()
        elif word == "add" and rest.split(" ", 1)[0] in PHI_TYPES and " " in rest:
            t, text = rest.split(" ", 1)
            add.append({"text": text, "type": t})
        else:
            print("  ? try: unmark p3   or   add NAME Tan Ah Kow")

    done = pipeline.confirm_deid(r["deid_id"], unmark=unmark, add=add, use_spark=args.spark, show_text=True)
    if done["next"] == "hold":
        print("\nHold rule: these look like names (all ticked to hide):")
        show(done["hold"])
        keep = ask("ids that are NOT names (Enter = hide all) > ").split()
        done = pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": keep}, use_spark=args.spark, show_text=True)
    if done["next"] == "flags":
        print("\nMuse Spark found more (all ticked to hide):")
        show(done["flags"])
        keep = ask("ids that are NOT identifiers (Enter = hide all) > ").split()
        done = pipeline.confirm_deid_flags(r["deid_id"], keep=keep)

    print("\n--- Tagged report ---\n" + done["text"])
    print(f"\nMuse Spark: {done['layer2']}   summary: {done['summary']}")
    for w in done["warnings"]:
        print("WARNING:", w)
    print("\nSteps:")
    for t in done["trace"]:
        print(f"  {'ok ' if t['ok'] else '!! '} {t['step']:7} {t['observe']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
