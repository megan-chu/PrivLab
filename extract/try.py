"""
try.py - try extraction end to end on dev/extra reports in the terminal.

De-identifies each report locally (no name-check call), defines the field if the test pack
does not have it yet (1 live call; the draft is shown and accepted), then finds candidates
(1 live call per report). Uses a separate test pack and score log (runs/try_pack.yaml,
runs/try_score_log.jsonl), so your real pack and real score log are untouched.

With --teach (dev reports only) you confirm each result: Enter = accept the top answer (or
"not stated" when nothing was found), a number = pick that candidate, "v 2.1 | <evidence words>"
= type the value, "n" = not in this report. Each answer is scored and learned before the next
report, so the next prompt carries your examples. Reports already taught are skipped.

Usage:
  python -m extract.try PP0083 PP0185 PP0001 --field "tumour size"
  python -m extract.try PP0083 --field "tumour size" --effort low
  python -m extract.try PP0083 PP0185 PP0001 PP0041 PP0110 --field "tumour size" --teach
  python -m eval.score_fields --phase teach --log runs/try_score_log.jsonl
"""
import argparse
import re
import sys

from common.config import ROOT
from common.data import load_phi_truth, load_report
from deid import pipeline
from deid.profile import new_profile
from extract.candidates import find_candidates
from extract.define import define_field
from extract.learn import confirm_field, read_log
from extract.pack import load_pack

TRY_PACK = ROOT / "runs" / "try_pack.yaml"
TRY_LOG = ROOT / "runs" / "try_score_log.jsonl"
PROMPT = "  your answer  (Enter = accept  ·  2 = pick #2  ·  v 2.1 | <evidence words>  ·  n = not stated) > "
TYPED = re.compile(r"^v\s*(?P<value>[^|]+?)\s*\|\s*(?P<evidence>.+)$", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Try extraction (live Muse Spark calls).")
    ap.add_argument("report_ids", nargs="+")
    ap.add_argument("--field", required=True, help='plain words, e.g. "tumour size"')
    ap.add_argument("--description", default=None)
    ap.add_argument("--effort", default=None, help="reasoning effort for find_candidates")
    ap.add_argument("--teach", action="store_true", help="confirm each result (dev only); scored and learned")
    args = ap.parse_args()

    for rid in args.report_ids:
        split = load_phi_truth(rid)["split"]
        if split == "test" or (args.teach and split != "dev"):
            print(f"{rid} is in the {split} split; {'teach on dev only' if args.teach else 'test is refused'}.")
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

    taught = {e["report_id"] for e in read_log(TRY_LOG) if e["field"] == key} if args.teach else set()
    for rid in args.report_ids:
        if rid in taught:
            print(f"\n=== {rid}: already taught for '{key}'; skipped (each report is learned once).")
            continue
        r = pipeline.deidentify_report(text=load_report(rid), report_id=rid, profile=new_profile()[0])
        done = pipeline.confirm_deid(r["deid_id"], use_spark=False)
        if done["next"] == "hold":
            pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=False)
        res = find_candidates(r["deid_id"], key, pack_path=TRY_PACK, effort=args.effort)
        if res["status"] == "error":
            print(f"\n=== {rid}: ERROR  {res['reason']}")
            continue
        print(f"\n=== {rid}: {res['status'].upper()}  value={res['value']} {res['unit'] or ''}  "
              f"section={res['section']}  reason={res['reason']}")
        for c in res["candidates"]:
            failed = [k for k, ok in c["checks"].items() if not ok]
            print(f"  {c['rank']}. [{c['id']}] {c['value']!s:6} conf={c['confidence']:<4} section={c['section']!s:11} "
                  f"failed={failed or '-'}  \"{c['evidence_text'][:70]}\"")
        if args.teach:
            print(teach(res, key))
    return 0


def teach(res: dict, key: str) -> str:
    """Ask the user for the right answer, then confirm_field (scored, then learned)."""
    def confirm(**kw):
        return confirm_field(res["deid_id"], key, pack_path=TRY_PACK, log_path=TRY_LOG, **kw)

    while True:
        try:
            ans = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nstopped")
        typed = TYPED.match(ans)
        if ans == "" and res["status"] == "missing":
            out = confirm(missing=True)                       # accept "not stated"
        elif ans == "" and res["candidate_id"]:
            out = confirm(candidate_id=res["candidate_id"])
        elif ans == "":
            print("  ! no top answer to accept: pick a number, type the value (v ... | ...), or n")
            continue
        elif ans.isdecimal() and 1 <= int(ans) <= len(res["candidates"]):
            out = confirm(candidate_id=res["candidates"][int(ans) - 1]["id"])
        elif ans.lower() == "n":
            out = confirm(missing=True)
        elif typed:
            out = confirm(value=typed["value"], evidence_text=typed["evidence"])
        else:
            print(f"  ? e.g. Enter, 1-{len(res['candidates']) or 1}, n, or: v 2.1 | tumour measures 2.1 cm")
            continue
        if out["status"] == "saved":
            return f"  -> {out['outcome']}; readiness now {out['readiness']}"
        print(f"  ! {out['reason']}")


if __name__ == "__main__":
    sys.exit(main())
