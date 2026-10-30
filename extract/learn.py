"""
learn.py - confirm_field and learning from it, scored "test then learn" (PLAN.md §5).

  1. find_candidates keeps its prediction unchanged (remember_prediction).
  2. confirm_field writes prediction vs the user's answer to runs/score_log.jsonl:
     correct / wrong / missed / extra / true_empty.
  3. Only then the pack learns: the confirmed span becomes a good example, a wrong top pick a
     bad one (last 3 + 3 kept).
  4. validation(): per field and definition version. The error that matters is a silent one: a green
     value that was wrong. Ready = with 95% confidence at most 5% of green values are wrong (exact
     binomial bound: 59 green values with no error, 93 with one, 124 with two). Blind checks (the
     answer hidden from the user) are counted separately, to show whether confirmations are anchored.

The log stores values, report ids and evidence positions only, never report text.
"""
import json
import time
from math import comb
from datetime import datetime, timedelta, timezone

from common.config import ROOT
from common.data import manifest
from common.timing import record
from deid import pipeline
from deid.names import locate
from deid.patterns import leftover_identifiers
from extract.checks import parse_number
from extract.pack import get_field, load_pack, put_field, save_pack
from extract.sections import section_at

LOG_PATH = ROOT / "runs" / "score_log.jsonl"
SGT = timezone(timedelta(hours=8))
KEEP_EXAMPLES = 3
CONTEXT = 60                          # characters of de-identified text around an example
_PREDICTIONS: dict[tuple[str, str], dict] = {}
_SHOWN_AT: dict[tuple[str, str], float] = {}     # when the result was shown to the user


def remember_prediction(result: dict) -> None:
    """Called by find_candidates: freeze what PrivLab predicted, before any feedback."""
    _PREDICTIONS[(result["deid_id"], result["field"])] = json.loads(json.dumps(result))
    _SHOWN_AT[(result["deid_id"], result["field"])] = time.monotonic()


def confirm_field(deid_id: str, field_key: str, candidate_id: str | None = None, value=None,
                  evidence_text: str | None = None, missing: bool = False, note: str | None = None, *,
                  phase: str = "teach", pack_path=None, log_path=None) -> dict:
    """The user's answer: pick a candidate, or give value + evidence, or say it is not in the report."""
    predicted = _PREDICTIONS.get((deid_id, field_key))
    result = pipeline.result_for(deid_id)
    pack = load_pack(pack_path)
    field = get_field(pack, field_key)
    if predicted is None or result is None or field is None:
        return {"status": "error", "reason": "run find_candidates for this report and field first"}
    if predicted.get("blind") and candidate_id:
        return {"status": "error", "reason": "blind check: give value with evidence_text, or missing=True"}

    confirmed = _confirmed(predicted, result.text, field, candidate_id, value, evidence_text, missing)
    if isinstance(confirmed, str):
        return {"status": "error", "reason": confirmed}
    outcome = _outcome(predicted["value"], confirmed["value"] if confirmed else None, field.type)
    review_s = round(time.monotonic() - _SHOWN_AT.get((deid_id, field_key), time.monotonic()), 1)

    # 1. Score first, with the prediction exactly as it was.
    entry = {"ts": datetime.now(SGT).isoformat(timespec="seconds"), "deid_id": deid_id,
             "report_id": result.report_id, "split": _split(result.report_id), "phase": phase,
             "field": field.key, "field_version": field.version, "blind": bool(predicted.get("blind")),
             "pack_version": pack["version"],
             "predicted": {k: predicted[k] for k in ("value", "status", "start", "end", "reason")},
             "confirmed": confirmed, "outcome": outcome,
             "model_ms": predicted.get("model_ms"), "review_s": review_s}       # for speed improvements

    # 2. Then learn.
    t0 = time.perf_counter()
    _learn(field, predicted, confirmed, result.text, outcome, result.kept_types, note)
    path = log_path or LOG_PATH
    field.readiness = readiness(validation(field, read_log(path) + [entry]))
    put_field(pack, field)
    save_pack(pack, pack_path)
    entry["learn_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    entry["readiness_after"] = field.readiness

    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    _PREDICTIONS.pop((deid_id, field_key), None)
    _SHOWN_AT.pop((deid_id, field_key), None)
    record("field_review", review_s * 1000, report_id=result.report_id, field=field.key,
           status=predicted["status"], outcome=outcome)
    record("field_learn", entry["learn_ms"], report_id=result.report_id, field=field.key)
    out = {"status": "saved", "outcome": outcome, "value": confirmed["value"] if confirmed else None,
           "start": confirmed["start"] if confirmed else None, "end": confirmed["end"] if confirmed else None,
           "readiness": field.readiness}              # start/end: the confirmed quote, in the de-identified text
    if entry["blind"]:                                 # revealed only after the user answered
        out.update(blind=True, predicted=predicted["value"])
    return out


def _confirmed(predicted, text, field, candidate_id, value, evidence_text, missing):
    if missing:
        return None
    if candidate_id:
        c = next((c for c in predicted["candidates"] if c["id"] == candidate_id), None)
        if c is None or c["start"] is None:
            return f"candidate {candidate_id} not found or has no evidence in the report"
        if c["fail"] is not None or c["value"] is None:
            return f"candidate {candidate_id} failed the {c['fail']} check; type the value instead"
        return {"value": c["value"], "start": c["start"], "end": c["end"]}
    if value is None or str(value).strip() == "" or not evidence_text:
        return "give candidate_id, or value with evidence_text, or missing=True"
    hits = locate(text, evidence_text)
    if not hits:
        return "evidence_text not found in the de-identified report"
    if field.type == "number":
        typed, value = value, parse_number(str(value), None, field.unit)
        if value is None:
            return f"'{typed}' is not a number; for a number field type e.g. 2.1"
    elif not any(c.lower() == str(value).strip().lower() for c in field.choices):
        return f"'{value}' is not one of: {', '.join(field.choices)}"
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


def _where(section: str | None) -> str:
    return f"the {section.replace('_', ' ')} section" if section and section != "other" else "the report"


def critique(field, top: dict, confirmed: dict | None, text: str, note: str | None = None) -> str:
    """Why a rejected pick was wrong, for the "do not pick" example: written by code from what it
    knows (both values, both sections), plus the reviewer's own words if given. No model call."""
    unit = f" {field.unit}" if field.type == "number" and field.unit else ""
    fmt = lambda v: f"{v:g}" if isinstance(v, (int, float)) else str(v)
    picked = f"{fmt(top['value'])}{unit}" if top.get("value") is not None else "this"
    where_a = _where(top.get("section") or section_at(text, top["start"]))
    if confirmed is None:
        why = f"Picked {picked} from {where_a}, but this report does not state {field.name}."
    else:
        right = f"{fmt(confirmed['value'])}{unit}"
        where_b = _where(section_at(text, confirmed["start"]))
        why = (f"Picked {picked} from {where_a}; the right value was {right} in {where_b}." if where_a != where_b
               else f"Picked {picked}; the right value was {right}, elsewhere in {where_b}.")
    if note and note.strip():
        why += f" Reviewer: {' '.join(note.split())[:200]}"
    return why


def _learn(field, predicted, confirmed, text: str, outcome: str, kept_types=(), note: str | None = None) -> None:
    def context(start, end):
        return " ".join(text[max(0, start - CONTEXT):end + CONTEXT].split())

    def safe(snippet: str) -> bool:
        # Examples travel with every later prompt: never keep one with an identifier still in it.
        return not leftover_identifiers(snippet, kept_types)

    if confirmed and safe(ctx := context(confirmed["start"], confirmed["end"])):
        good = {"report": predicted["deid_id"], "value": confirmed["value"], "context": ctx}
        field.examples["good"] = (field.examples["good"] + [good])[-KEEP_EXAMPLES:]
    top = next((c for c in predicted["candidates"] if c["id"] == predicted["candidate_id"]), None)
    if outcome in ("wrong", "extra") and top and top["start"] is not None and safe(ctx := context(top["start"], top["end"])):
        reason = critique(field, top, confirmed, text, note)
        if not safe(reason):                                   # a reviewer note with an identifier
            reason = critique(field, top, confirmed, text)
        bad = {"report": predicted["deid_id"], "context": ctx, "reason": reason}
        field.examples["bad"] = (field.examples["bad"] + [bad])[-KEEP_EXAMPLES:]

    field.recent = (field.recent + ["accepted" if outcome in ("correct", "true_empty") else "corrected"])[-5:]


# --------------------------------------------------------------------------
# Validation: per field and definition version, from the log
# --------------------------------------------------------------------------
TARGET = 0.05             # at most 5% of green values wrong ...
CONFIDENCE = 0.95         # ... with 95% confidence


def upper_bound(errors: int, n: int) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on an error rate, after `errors` in `n`."""
    if n == 0 or errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(50):
        p = (lo + hi) / 2
        at_most = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(errors + 1))
        lo, hi = (p, hi) if at_most > 1 - CONFIDENCE else (lo, p)
    return hi


def more_needed(errors: int, n: int) -> int:
    """How many more correct green values would bring the bound to TARGET."""
    if upper_bound(errors, n) <= TARGET:
        return 0
    hi = 1
    while upper_bound(errors, n + hi) > TARGET:
        hi *= 2
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if upper_bound(errors, n + mid) > TARGET else (lo, mid)
    return hi


def validation(field, entries: list[dict]) -> dict:
    """Scores for the field's current definition version only."""
    rows = [e for e in entries if e["field"] == field.key and e.get("field_version", 1) == field.version]
    green = [e for e in rows if e["predicted"]["status"] == "green"]
    silent = sum(e["outcome"] in ("wrong", "extra") for e in green)
    blind = [e for e in rows if e.get("blind")]
    right = lambda es: sum(e["outcome"] in ("correct", "true_empty") for e in es)
    bound = upper_bound(silent, len(green))
    return {"version": field.version, "checked": len(rows), "correct": right(rows),
            "green": len(green), "silent_errors": silent, "silent_bound": round(bound, 4),
            "more_needed": more_needed(silent, len(green)),
            "review_rate": round(1 - len(green) / len(rows), 2) if rows else None,
            "blind": len(blind), "blind_correct": right(blind), "ready": bound <= TARGET}


def readiness(v: dict) -> str:
    return "draft" if not v["checked"] else "ready" if v["ready"] else "learning"


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
    log = read_log(log_path)
    fields = [{"key": f.key, "name": f.name, "type": f.type, "unit": f.unit, "choices": f.choices,
               "readiness": f.readiness, "recent": f.recent, "version": f.version,
               "validation": validation(f, log)} for f in pack["fields"]]
    if field_key is None:
        return {"version": pack["version"], "fields": fields}
    field = get_field(pack, field_key)
    if field is None:
        return {"status": "error", "reason": f"no field '{field_key}' in the pack"}
    extractions = [{"when": e["ts"], "report_id": e["report_id"], "version": e.get("field_version", 1),
                    "predicted": e["predicted"]["value"], "status": e["predicted"]["status"],
                    "confirmed": (e["confirmed"] or {}).get("value"), "outcome": e["outcome"],
                    "blind": e.get("blind", False), "readiness_after": e.get("readiness_after")}
                   for e in reversed(log) if e["field"] == field_key]
    return {"field": {**next(f for f in fields if f["key"] == field_key), "meaning": field.meaning,
                      "definition": field.definition, "description": field.description, "range": field.range},
            "definition_history": field.history, "extraction_history": extractions}
