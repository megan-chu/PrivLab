"""
learn.py - confirm_field and learning from it, scored "test then learn" (PLAN.md §5).

  1. find_candidates keeps its prediction unchanged (remember_prediction).
  2. confirm_field writes prediction vs the user's answer to runs/score_log.jsonl:
     correct / wrong / missed / extra / true_empty.
  3. Only then the pack learns: the confirmed span becomes a good example, a wrong top pick a
     bad one (last 3 + 3 kept), and readiness moves Draft -> Learning -> Ready
     (Ready = at least 4 of the last 5 accepted unchanged).

The log stores values, report ids and evidence positions only, never report text.
"""
import json
from datetime import datetime, timedelta, timezone

from common.config import ROOT
from common.data import manifest
from deid import pipeline
from deid.names import locate
from extract.checks import parse_number
from extract.pack import get_field, load_pack, put_field, save_pack

LOG_PATH = ROOT / "runs" / "score_log.jsonl"
SGT = timezone(timedelta(hours=8))
KEEP_EXAMPLES = 3
CONTEXT = 60                          # characters of de-identified text around an example
_PREDICTIONS: dict[tuple[str, str], dict] = {}


def remember_prediction(result: dict) -> None:
    """Called by find_candidates: freeze what PrivLab predicted, before any feedback."""
    _PREDICTIONS[(result["deid_id"], result["field"])] = json.loads(json.dumps(result))


def confirm_field(deid_id: str, field_key: str, candidate_id: str | None = None, value=None,
                  evidence_text: str | None = None, missing: bool = False, *,
                  phase: str = "teach", pack_path=None, log_path=None) -> dict:
    """The user's answer: pick a candidate, or give value + evidence, or say it is not in the report."""
    predicted = _PREDICTIONS.get((deid_id, field_key))
    result = pipeline.result_for(deid_id)
    pack = load_pack(pack_path)
    field = get_field(pack, field_key)
    if predicted is None or result is None or field is None:
        return {"status": "error", "reason": "run find_candidates for this report and field first"}

    confirmed = _confirmed(predicted, result.text, field, candidate_id, value, evidence_text, missing)
    if isinstance(confirmed, str):
        return {"status": "error", "reason": confirmed}
    outcome = _outcome(predicted["value"], confirmed["value"] if confirmed else None, field.type)

    # 1. Score first, with the prediction exactly as it was.
    entry = {"ts": datetime.now(SGT).isoformat(timespec="seconds"), "deid_id": deid_id,
             "report_id": result.report_id, "split": _split(result.report_id), "phase": phase,
             "field": field.key, "pack_version": pack["version"],
             "predicted": {k: predicted[k] for k in ("value", "status", "start", "end", "reason")},
             "confirmed": confirmed, "outcome": outcome}

    # 2. Then learn.
    _learn(field, predicted, confirmed, result.text, outcome)
    put_field(pack, field)
    save_pack(pack, pack_path)
    entry["readiness_after"] = field.readiness

    path = log_path or LOG_PATH
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    _PREDICTIONS.pop((deid_id, field_key), None)
    return {"status": "saved", "outcome": outcome, "value": confirmed["value"] if confirmed else None,
            "readiness": field.readiness}


def _confirmed(predicted, text, field, candidate_id, value, evidence_text, missing):
    if missing:
        return None
    if candidate_id:
        c = next((c for c in predicted["candidates"] if c["id"] == candidate_id), None)
        if c is None or c["start"] is None:
            return f"candidate {candidate_id} not found or has no evidence in the report"
        return {"value": c["value"], "start": c["start"], "end": c["end"]}
    if value is None or not evidence_text:
        return "give candidate_id, or value with evidence_text, or missing=True"
    hits = locate(text, evidence_text)
    if not hits:
        return "evidence_text not found in the de-identified report"
    if field.type == "number":
        value = parse_number(str(value), None, field.unit)
    return {"value": value, "start": hits[0][0], "end": hits[0][1]}


def _same(a, b, kind: str) -> bool:
    if kind == "number":
        return abs(float(a) - float(b)) < 0.05
    return str(a).strip().lower() == str(b).strip().lower()


def _outcome(pred, conf, kind: str) -> str:
    if pred is None and conf is None:
        return "true_empty"
    if pred is None:
        return "missed"
    if conf is None:
        return "extra"
    return "correct" if _same(pred, conf, kind) else "wrong"


def _learn(field, predicted, confirmed, text: str, outcome: str) -> None:
    def context(start, end):
        return " ".join(text[max(0, start - CONTEXT):end + CONTEXT].split())

    if confirmed:
        good = {"report": predicted["deid_id"], "value": confirmed["value"],
                "context": context(confirmed["start"], confirmed["end"])}
        field.examples["good"] = (field.examples["good"] + [good])[-KEEP_EXAMPLES:]
    top = next((c for c in predicted["candidates"] if c["id"] == predicted["candidate_id"]), None)
    if outcome in ("wrong", "extra") and top and top["start"] is not None:
        bad = {"report": predicted["deid_id"], "context": context(top["start"], top["end"]),
               "reason": "not this one" if confirmed else "not stated in this report"}
        field.examples["bad"] = (field.examples["bad"] + [bad])[-KEEP_EXAMPLES:]

    field.recent = (field.recent + ["accepted" if outcome in ("correct", "true_empty") else "corrected"])[-5:]
    accepted = field.recent.count("accepted")
    field.readiness = "ready" if len(field.recent) >= 5 and accepted >= 4 else "learning"


def _split(report_id: str | None) -> str | None:
    return next((r["split"] for r in manifest() if r["report_id"] == report_id), None)


# --------------------------------------------------------------------------
# Field history (show_pack)
# --------------------------------------------------------------------------
def read_log(log_path=None) -> list[dict]:
    path = log_path or LOG_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def show_pack(field_key: str | None = None, *, pack_path=None, log_path=None) -> dict:
    """All fields with readiness; with field_key, also its definition and extraction history."""
    pack = load_pack(pack_path)
    fields = [{"key": f.key, "name": f.name, "type": f.type, "unit": f.unit, "choices": f.choices,
               "readiness": f.readiness, "recent": f.recent} for f in pack["fields"]]
    if field_key is None:
        return {"version": pack["version"], "fields": fields}
    field = get_field(pack, field_key)
    if field is None:
        return {"status": "error", "reason": f"no field '{field_key}' in the pack"}
    extractions = [{"when": e["ts"], "report_id": e["report_id"], "predicted": e["predicted"]["value"],
                    "status": e["predicted"]["status"], "confirmed": (e["confirmed"] or {}).get("value"),
                    "outcome": e["outcome"], "readiness_after": e.get("readiness_after")}
                   for e in reversed(read_log(log_path)) if e["field"] == field_key]
    return {"field": {**next(f for f in fields if f["key"] == field_key), "meaning": field.meaning,
                      "description": field.description, "range": field.range},
            "definition_history": field.history, "extraction_history": extractions}
