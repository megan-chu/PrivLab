"""
try.py - try extraction end to end on dev/extra reports in the terminal.

De-identifies each report locally (no name-check call), defines the field if the test pack
does not have it yet (1 live call; the draft is shown and accepted), then finds candidates
(1 live call per report). Uses a separate test pack, runs/try_pack.yaml, so your real pack
is untouched.

With --teach you confirm each result (Enter = accept, a number = pick that candidate,
"v 2.1 | <evidence words>" = type the value, "n" = not in this report); each answer is scored
and learned before the next report, so the next prompt carries your examples.

Usage:
  python -m extract.try PP0083 PP0185 PP0001 --field "tumour size"
  python -m extract.try PP0083 --field "tumour size" --effort low
  python -m extract.try PP0083 PP0185 PP0001 PP0041 PP0110 --field "tumour size" --teach
"""
import argparse
import sys

from common.config import ROOT
from common.data import load_phi_truth, load_report
from deid import pipeline
from deid.profile import new_profile
from extract.candidates import find_candidates
from extract.define import define_field
from extract.pack import load_pack

TRY_PACK = ROOT / "runs" / "try_pack.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="Try extraction (live Muse Spark calls).")
    ap.add_argument("report_ids", nargs="+")
    ap.add_argument("--field", required=True, help='plain words, e.g. "tumour size"')
    ap.add_argument("--description", default=None)
    ap.add_argument("--effort", default=None, help="reasoning effort for find_candidates")
    ap.add_argument("--teach", action="store_true", help="confirm each result; it is scored and learned")
    args = ap.parse_args()

    for rid in args.report_ids:
        if load_phi_truth(rid)["split"] == "test":
            print(f"{rid} is in the test split; refused.")
            return 2

    pack = load_pack(TRY_PACK)
    field = next((f for f in pack["fields"] if f.name == args.field), None)
    if field is None:
        draft = define_field(args.field, args.description, pack_path=TRY_PACK)
        print(f"\nDraft: {draft['draft']}")
        field = define_field(args.field, args.description, accept=draft["draft"], pack_path=TRY_PACK)["field"]
        print(f"Saved field '{field['key']}' to {TRY_PACK.relative_to(ROOT)}")
        key = field["key"]
    else:
        key = field.key

    for rid in args.report_ids:
        r = pipeline.deidentify_report(text=load_report(rid), report_id=rid, profile=new_profile()[0])
        done = pipeline.confirm_deid(r["deid_id"], use_spark=False)
        if done["next"] == "hold":
            pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=False)
        res = find_candidates(r["deid_id"], key, pack_path=TRY_PACK, effort=args.effort)
        print(f"\n=== {rid}: {res['status'].upper()}  value={res['value']} {res['unit'] or ''}  "
              f"section={res['section']}  reason={res['reason']}")
        for c in res["candidates"]:
            failed = [k for k, ok in c["checks"].items() if not ok]
            print(f"  {c['rank']}. [{c['id']}] {c['value']!s:6} conf={c['confidence']:<4} section={c['section']!s:11} "
                  f"failed={failed or '-'}  \"{c['evidence_text'][:70]}\"")
        if args.teach and res["status"] != "error":
            print(teach(res, key))
    return 0


def teach(res: dict, key: str) -> str:
    """Ask the user for the right answer, then confirm_field (scored, then learned)."""
    from extract.learn import confirm_field

    while True:
        try:
            ans = input("  your answer [Enter=accept | 2 = pick #2 | v 2.1 | evidence words | n = not stated] > ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nstopped")
        if ans == "" and res["candidate_id"]:
            out = confirm_field(res["deid_id"], key, candidate_id=res["candidate_id"], pack_path=TRY_PACK)
        elif ans.isdigit() and int(ans) <= len(res["candidates"]):
            out = confirm_field(res["deid_id"], key, candidate_id=res["candidates"][int(ans) - 1]["id"],
                                pack_path=TRY_PACK)
        elif ans.lower() == "n":
            out = confirm_field(res["deid_id"], key, missing=True, pack_path=TRY_PACK)
        elif ans.startswith("v ") and "|" in ans:
            value, evidence = (s.strip() for s in ans[2:].split("|", 1))
            out = confirm_field(res["deid_id"], key, value=value, evidence_text=evidence, pack_path=TRY_PACK)
        else:
            print("  ? e.g. Enter, 2, n, or: v 2.1 | tumour measures 2.1 cm")
            continue
        if out["status"] == "saved":
            return f"  -> {out['outcome']}; readiness now {out['readiness']}"
        print(f"  ! {out['reason']}")


if __name__ == "__main__":
    sys.exit(main())
