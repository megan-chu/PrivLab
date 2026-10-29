"""
candidates.py - find_candidates: likely values for a pack field in a de-identified report,
each with evidence, checks and a reason; best first; green, amber or missing.

One Muse Spark call per report covers every field in the pack (cached), and sees only the
sealed DeidResult text. The model proposes; code finds the evidence, runs the checks, detects
the section and decides the status.

  find_candidates(deid_id, field_key) -> FieldResult dict
"""
from common import spark
from deid import pipeline
from extract.checks import run_checks
from extract.pack import get_field, load_pack
from extract.sections import PRIORITY, WEAK
from extract.types import CandidatesReply

INSTRUCTIONS = (
    "You extract fields from a de-identified pathology report for a cancer registry. Identifiers are "
    "tags like [PATIENT-1]. For each field below, list every place the report states a value, best first. "
    "value_text and unit_text: exactly as written in the report. evidence_text: the exact phrase or "
    "sentence, copied character for character. confidence: 0 to 1. If a field is not stated, return an "
    "empty candidates list. Do not infer, calculate or give medical advice."
)
MIN_CONFIDENCE = 0.6


def _field_spec(f) -> str:
    spec = f"- key: {f.key}\n  meaning: {f.meaning}\n  type: {f.type}"
    if f.type == "number":
        spec += f", unit {f.unit or '-'}" + (f", range {f.range[0]}-{f.range[1]}" if f.range else "")
    else:
        spec += f", one of: {', '.join(f.choices)}"
    for ex in f.examples.get("good", [])[-3:]:
        spec += f"\n  correct example: \"{ex['context']}\" -> {ex['value']}"
    for ex in f.examples.get("bad", [])[-3:]:
        spec += f"\n  wrong example (do not pick): \"{ex['context']}\" ({ex.get('reason', '')})"
    return spec


def _reply_for(result, pack) -> CandidatesReply:
    fields = "\n".join(_field_spec(f) for f in pack["fields"])
    return spark.call(result, instructions=f"{INSTRUCTIONS}\n\nFields:\n{fields}", schema=CandidatesReply,
                      step="candidates", report_id=result.report_id)


def find_candidates(deid_id: str, field_key: str, *, pack_path=None) -> dict:
    result = pipeline.result_for(deid_id)
    if result is None:
        return {"deid_id": deid_id, "status": "error", "reason": "de-identify and confirm this report first"}
    pack = load_pack(pack_path)
    field = get_field(pack, field_key)
    if field is None:
        return {"deid_id": deid_id, "status": "error", "reason": f"no field '{field_key}' in the pack; define it first"}

    try:
        reply = _reply_for(result, pack)
    except Exception as e:   # API down, timeout, bad JSON: amber, never a crash
        return _result(deid_id, field, "amber", None, [], f"Muse Spark unavailable ({type(e).__name__}); review manually")

    proposed = next((fc.candidates for fc in reply.fields if fc.key == field.key), [])
    cands = []
    for i, c in enumerate(proposed, 1):
        checked = run_checks(field, c, result.text, result.kept_types)
        cands.append({"id": f"c{i}", "value": checked["value"], "unit": field.unit if field.type == "number" else None,
                      "evidence_text": c.evidence_text, "start": checked["start"], "end": checked["end"],
                      "section": checked["section"], "reason": c.reason, "confidence": round(c.confidence, 2),
                      "checks": checked["checks"], "fail": checked["fail"]})

    valid = [c for c in cands if c["fail"] is None]
    valid.sort(key=lambda c: (PRIORITY.get(c["section"], 2), c["confidence"]), reverse=True)
    ranked = valid + [c for c in cands if c["fail"] is not None]
    for rank, c in enumerate(ranked, 1):
        c["rank"] = rank

    if not cands:
        return _result(deid_id, field, "missing", None, ranked, "Not stated in this report")
    if not valid:
        return _result(deid_id, field, "amber", None, ranked, f"No candidate passed the checks ({ranked[0]['fail']})")
    top = valid[0]
    rivals = [c for c in valid[1:] if c["value"] != top["value"]
              and PRIORITY.get(c["section"], 2) == PRIORITY.get(top["section"], 2)]
    if rivals:
        reason = f"Two different values in the same section: {top['value']} vs {rivals[0]['value']}. Which one?"
    elif top["section"] in WEAK:
        reason = f"From the {top['section']} section only"
    elif top["confidence"] < MIN_CONFIDENCE:
        reason = "Low confidence"
    else:
        return _result(deid_id, field, "green", top, ranked, None)
    return _result(deid_id, field, "amber", top, ranked, reason)


def _result(deid_id: str, field, status: str, top: dict | None, ranked: list, reason: str | None) -> dict:
    return {"deid_id": deid_id, "field": field.key, "status": status,
            "value": top["value"] if top else None, "unit": top["unit"] if top else None,
            "candidate_id": top["id"] if top else None, "evidence_text": top["evidence_text"] if top else None,
            "start": top["start"] if top else None, "end": top["end"] if top else None,
            "section": top["section"] if top else None, "reason": reason,
            "candidates": ranked, "readiness": field.readiness}
