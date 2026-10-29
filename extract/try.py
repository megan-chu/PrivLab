"""
try.py - try extraction end to end on dev/extra reports in the terminal.

De-identifies each report locally (no name-check call), defines the field if the test pack
does not have it yet (1 live call; the draft is shown and accepted), then finds candidates
(1 live call per report). Uses a separate test pack, runs/try_pack.yaml, so your real pack
is untouched.

Usage:
  python -m extract.try PP0083 PP0185 PP0001 --field "tumour size"
  python -m extract.try PP0083 --field "tumour size" --effort low
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
            print(f"  {c['rank']}. {c['value']!s:6} conf={c['confidence']:<4} section={c['section']!s:11} "
                  f"failed={failed or '-'}  \"{c['evidence_text'][:70]}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
