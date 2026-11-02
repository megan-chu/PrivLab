"""
learn.py - confirm_field and learning from it, scored "test then learn" (PLAN.md §5).

  1. find_candidates keeps its prediction unchanged (remember_prediction).
  2. confirm_field writes prediction vs the user's answer to runs/score_log.jsonl:
     correct / wrong / missed / extra / true_empty.
  3. Only then the pack learns: the confirmed span becomes a good example, a wrong top pick a
     bad one (last 3 + 3 kept).
  4. validation(): per field and definition version. The error that matters is a silent one: a green
     value that was wrong. Ready = with 95% confidence at most 5% of green values are wrong (exact
     binomial bound: 59 green values with no error, 93 with one, 124 with two). The bound is pooled over
     the pack's fields (one reviewed report checks them all), and a field with no silent error also needs
     MIN_FIELD green checks of its own, so a bad field cannot hide behind good ones. Blind checks (the
     answer hidden from the user) are counted separately, to show whether confirmations are anchored.
     Only a report's first check counts: checked again later, PrivLab has already learned from that report.
     "data" is the export's view: values a person confirmed (right by definition) and values saved by bulk.
  5. Answer memory: each entry keeps a fingerprint of the report (SHA-256 of the file's text, never the text). The
     same report again gets the person's earlier answer back (phase "remembered"): no model call, never scored or
     taught. A reset clears it.

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
from extract.checks import norm_value, parse_number, tidy_text
from extract.pack import get_field, load_pack, put_field, save_pack
from extract.sections import section_at

LOG_PATH = ROOT / "runs" / "score_log.jsonl"
SGT = timezone(timedelta(hours=8))
KEEP_EXAMPLES = 3
KEEP_RULES = 50           # learned corrections per field ("mammae sinistrae" -> left breast)
CONTEXT = 60                          # characters of de-identified text around an example
_PREDICTIONS: dict[tuple[str, str], dict] = {}
_SHOWN_AT: dict[tuple[str, str], float] = {}     # when the result was shown to the user


def remember_prediction(result: dict) -> None:
    """Called by find_candidates: freeze what PrivLab predicted, before any feedback."""
    _PREDICTIONS[(result["deid_id"], result["field"])] = json.loads(json.dumps(result))
    _SHOWN_AT[(result["deid_id"], result["field"])] = time.monotonic()


def confirm_field(deid_id: str, field_key: str, candidate_id: str | None = None, value=None,
                  evidence_text: str | None = None, missing: bool = False, note: str | None = None, *,
                  items: list[dict] | None = None, at: int | None = None, phase: str = "teach", pack_path=None,
                  log_path=None) -> dict:
    """The user's answer: pick a candidate, or give value + evidence, or say it is not in the report.
    at: where the user selected the evidence (character offset), so a quote that occurs twice lands on theirs.
    A list field takes items=[{"value", "evidence_text"}, ...] instead: the whole list, as confirmed.

    phase="bulk": a verified value accepted by "Extract all", not by a person. It is saved for the export but
    is no evidence of accuracy, so it never scores (validation, score_fields) and never teaches the pack.
    phase="remembered": the person's earlier answer on the same text, replayed: saved, never scored or taught."""
    predicted = _PREDICTIONS.get((deid_id, field_key))
    result = pipeline.result_for(deid_id)
    pack = load_pack(pack_path)
    field = get_field(pack, field_key)
    if predicted is None or result is None or field is None:
        return {"status": "error", "reason": "run find_candidates for this report and field first"}
    if predicted.get("blind") and candidate_id:
        return {"status": "error", "reason": "blind check: give value with evidence_text, or missing=True"}
    if phase == "bulk" and (predicted["status"] != "green" or predicted.get("blind")):
        return {"status": "error", "reason": "only a verified value can be accepted without review"}
    if (phase == "remembered") != (predicted["status"] == "remembered"):
        return {"status": "error", "reason": "only a remembered answer is saved as remembered"}
    if phase == "bulk" and not proven(field, pack["fields"], log_path):
        return {"status": "error", "reason": "this field is still learning: a person confirms its values until it is "
                                             "Proven accurate"}

    confirmed = (_confirmed_items(result.text, field, items, missing) if field.multi
                 else _confirmed(predicted, result.text, field, candidate_id, value, evidence_text, missing, at))
    if isinstance(confirmed, str):
        return {"status": "error", "reason": confirmed}
    outcome = _outcome(predicted["value"], confirmed["value"] if confirmed else None, field.type)
    review_s = round(time.monotonic() - _SHOWN_AT.get((deid_id, field_key), time.monotonic()), 1)

    # 1. Score first, with the prediction exactly as it was.
    entry = {"ts": datetime.now(SGT).isoformat(timespec="seconds"), "deid_id": deid_id,
             "report_id": result.report_id, "split": _split(result.report_id), "phase": phase,
             "text_hash": pipeline.fingerprint(deid_id),                    # the fingerprint, never the text
             "field": field.key, "field_version": field.version, "blind": bool(predicted.get("blind")),
             "pack_version": pack["version"],
             "predicted": {k: predicted[k] for k in ("value", "status", "start", "end", "reason")},
             "confirmed": confirmed, "outcome": outcome,
             "model_ms": predicted.get("model_ms"), "review_s": review_s}       # for speed improvements

    # 2. Then learn.
    t0 = time.perf_counter()
    if phase not in ("bulk", "remembered"):
        _learn(field, predicted, confirmed, result.text, outcome, result.kept_types, note)
    path = log_path or LOG_PATH
    log = read_log(path) + [entry]
    field.readiness = readiness(validation(field, log, pack["fields"]))
    counted = any(e is entry for e in _checks(field, log))       # a report checked before: it teaches, not scores
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
           "readiness": field.readiness, "counted": counted}   # start/end: the confirmed quote, in the de-identified text
    if field.multi:
        out["items"] = confirmed["items"] if confirmed else []
    if entry["blind"]:                                 # revealed only after the user answered
        out.update(blind=True, predicted=predicted["value"])
    return out


def _confirmed(predicted, text, field, candidate_id, value, evidence_text, missing, at=None):
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
    else:
        value = tidy_text(value)                                      # text: as written, minus slips
    start, end = min(hits, key=lambda h: abs(h[0] - at)) if isinstance(at, int) else hits[0]   # the one selected
    return {"value": value, "start": start, "end": end}


def reset_field(deid_id: str, field_key: str, *, pack_path=None, log_path=None) -> dict:
    """Undo a confirmed value (e.g. after the field was redefined). The earlier check stays in the log: it was a
    real answer and is still scored. A "reset" entry follows it, so the export no longer holds the value; resets
    never score and never teach. Read the field again (find_candidates) to answer it anew."""
    result = pipeline.result_for(deid_id)
    field = get_field(load_pack(pack_path), field_key)
    path = log_path or LOG_PATH
    latest = next((e for e in reversed(read_log(path)) if e["deid_id"] == deid_id and e["field"] == field_key), None)
    if result is None or field is None or latest is None or latest.get("phase") == "reset":
        return {"status": "error", "reason": "nothing confirmed to reset for this report and field"}
    entry = {"ts": datetime.now(SGT).isoformat(timespec="seconds"), "deid_id": deid_id, "report_id": result.report_id,
             "split": _split(result.report_id), "phase": "reset", "field": field.key, "field_version": field.version,
             "text_hash": pipeline.fingerprint(deid_id),
             "blind": False, "predicted": {"value": None, "status": "reset", "start": None, "end": None, "reason": None},
             "confirmed": None, "outcome": "reset"}
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    _PREDICTIONS.pop((deid_id, field_key), None)
    return {"status": "reset", "field": field.key}


def _confirmed_items(text, field, items, missing):
    """A list field: every item located in the text; value is the list of values, items keep their quotes."""
    if missing:
        return None
    if not items:
        return "give items=[{value, evidence_text}, ...], or missing=True"
    out = []
    for it in items:
        raw = " ".join(str(it.get("value") or "").split())
        value = tidy_text(raw)                                         # saved tidied; found as written
        hits = locate(text, it.get("evidence_text") or raw)
        if not value or not hits:
            return f"item '{value or '?'}' not found in the de-identified report"
        out.append({"value": value, "start": hits[0][0], "end": hits[0][1]})
    return {"value": [i["value"] for i in out], "start": out[0]["start"], "end": out[0]["end"], "items": out}


def _same(a, b, kind: str) -> bool:
    if kind == "number":
        return abs(float(a) - float(b)) < 0.05
    norm = lambda v: " ".join(str(v).lower().replace(".", " ").split())    # "ductal. carcinoma" = "ductal carcinoma"
    if isinstance(a, list) or isinstance(b, list):                        # a list: the same set of items
        return sorted(norm(x) for x in (a or [])) == sorted(norm(x) for x in (b or []))
    return norm(a) == norm(b)


def _outcome(pred, conf, kind: str) -> str:
    pred, conf = (pred or None if isinstance(pred, list) else pred), (conf or None if isinstance(conf, list) else conf)
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

    _learn_rule(field, predicted, top, confirmed, outcome)
    field.recent = (field.recent + ["accepted" if outcome in ("correct", "true_empty") else "corrected"])[-5:]


def _learn_rule(field, predicted, top, confirmed, outcome: str) -> None:
    """A correction of a text value becomes a rule: next time the model says the same words, code gives the user's
    value. A rule's first use asks the user; accepted once, it applies quietly; contradicted, it is dropped."""
    if field.type != "text" or field.multi or not top or top.get("value") is None:
        return
    rules = field.examples.setdefault("rules", [])
    raw = norm_value(top.get("raw", top["value"]))
    rule = next((r for r in rules if r["from"] == raw), None)
    if top.get("rule") and rule and outcome == "correct":
        rule["uses"] = rule.get("uses", 0) + 1                   # the user agreed with the rewrite
        return
    if outcome not in ("wrong", "extra"):
        return
    if rule:
        rules.remove(rule)                                     # contradicted: drop it
    if confirmed and confirmed.get("value") is not None and norm_value(confirmed["value"]) != raw:
        rules.append({"from": raw, "to": confirmed["value"], "report": predicted["deid_id"], "uses": 0})
        field.examples["rules"] = rules[-KEEP_RULES:]


# --------------------------------------------------------------------------
# Validation: per field and definition version, from the log
# --------------------------------------------------------------------------
TARGET = 0.05             # at most 5% of green values wrong ...
CONFIDENCE = 0.95         # ... with 95% confidence
MIN_FIELD = 10            # green checks a field needs of its own before the pooled bound can make it Ready
RECENT = 10               # first guesses on the latest reports, to show whether teaching is working


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


def _report(e: dict) -> str:
    return e.get("report_id") or e["deid_id"]


def _checks(field, entries: list[dict]) -> list[dict]:
    """A person's first check of each report, on the field's current definition version."""
    seen, out = set(), []
    for e in entries:
        if (e["field"] == field.key and e.get("field_version", 1) == field.version
                and e.get("phase") not in ("bulk", "reset", "remembered") and _report(e) not in seen):   # not a check
            seen.add(_report(e))
            out.append(e)
    return out


def _data(field, entries: list[dict]) -> dict:
    """The field's values as they stand in the export: confirmed by you (since any reset), or saved by bulk only."""
    by, first = {}, {}
    for e in entries:
        if e["field"] != field.key:
            continue
        r = _report(e)
        if e.get("phase") == "reset":
            by.pop(r, None)
            continue
        if e.get("phase") != "bulk":
            first.setdefault(r, e["outcome"])                  # its first check: did you have to correct it?
        if e.get("phase") != "bulk":
            by[r] = "yours"
        else:
            by.setdefault(r, "auto")                           # bulk never takes over a value you confirmed
    yours = [r for r, w in by.items() if w == "yours"]
    return {"yours": len(yours), "corrected": sum(first.get(r) not in ("correct", "true_empty") for r in yours),
            "auto": len(by) - len(yours)}


def _silent(rows: list[dict]) -> tuple[int, int]:
    green = [e for e in rows if e["predicted"]["status"] == "green"]
    return sum(e["outcome"] in ("wrong", "extra") for e in green), len(green)


def validation(field, entries: list[dict], pool=None) -> dict:
    """Scores for the field's current definition version only. pool: the pack's fields, for the pooled bound."""
    rows = _checks(field, entries)
    silent, green = _silent(rows)
    blind = [e for e in rows if e.get("blind")]
    right = lambda es: sum(e["outcome"] in ("correct", "true_empty") for e in es)
    bound, need = upper_bound(silent, green), more_needed(silent, green)
    pool = [f for f in (pool or []) if f.key != field.key]
    if pool and silent == 0:                          # pooled: every field's checks count towards one bound
        p_silent, p_green = map(sum, zip((silent, green), *(_silent(_checks(f, entries)) for f in pool)))
        bound = min(bound, upper_bound(p_silent, p_green) if green >= MIN_FIELD else 1.0)
        per_report = -(-more_needed(p_silent, p_green) // (len(pool) + 1))     # each report checks every field
        need = min(need, max(MIN_FIELD - green, per_report, 0))
    count = lambda *o: sum(e["outcome"] in o for e in rows)
    recent = rows[-RECENT:]
    return {"version": field.version, "checked": len(rows), "correct": right(rows),
            "outcomes": {"right": right(rows), "corrected": count("wrong"), "missed": count("missed"),
                         "extra": count("extra")},
            "recent": len(recent), "recent_correct": right(recent), "data": _data(field, entries),
            "green": green, "silent_errors": silent, "silent_bound": round(bound, 4),
            "more_needed": need if bound > TARGET else 0,
            "review_rate": round(1 - green / len(rows), 2) if rows else None,
            "blind": len(blind), "blind_correct": right(blind), "ready": bound <= TARGET}


def remembered(fingerprint: str | None, field, log_path=None) -> dict | None:
    """The person's latest answer for this field on exactly this report (its fingerprint; same definition
    version), unless they reset it since. A value Extract all saved alone is not an answer."""
    h, mem = fingerprint, None
    if not h:
        return None
    for e in read_log(log_path):
        if e.get("text_hash") != h or e["field"] != field.key or e.get("field_version", 1) != field.version:
            continue
        if e.get("phase") == "reset":
            mem = None
        elif e.get("phase") != "bulk":
            mem = e
    return mem


def proven(field, fields, log_path=None) -> bool:
    """Proven accurate (pooled bound): only then may a verified value be saved without a person."""
    return validation(field, read_log(log_path), fields)["ready"]


def readiness(v: dict) -> str:
    return "draft" if not v["checked"] else "ready" if v["ready"] else "learning"


def _split(report_id: str | None) -> str | None:
    return next((r["split"] for r in manifest() if r["report_id"] == report_id), None)


# --------------------------------------------------------------------------
# Field history (show_pack)
# --------------------------------------------------------------------------
def saved_for(deid_id: str, log_path=None) -> dict:
    """A report's confirmed values, latest per field (a reset field is left out): to reopen the report as it was."""
    latest = {}
    for e in read_log(log_path):
        if e["deid_id"] == deid_id:
            latest[e["field"]] = e
    out = {}
    for key, e in latest.items():
        if e.get("phase") == "reset":
            continue
        c = e.get("confirmed") or {}
        out[key] = {"status": "saved", "outcome": e["outcome"], "value": c.get("value"), "start": c.get("start"),
                    "end": c.get("end"), **({"items": c["items"]} if "items" in c else {})}
    return out


def read_log(log_path=None) -> list[dict]:
    path = log_path or LOG_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def show_pack(field_key: str | None = None, *, pack_path=None, log_path=None) -> dict:
    """All fields with readiness; with field_key, also its definition and extraction history."""
    pack = load_pack(pack_path)
    log = read_log(log_path)
    fields = [{"key": f.key, "name": f.name, "type": f.type, "unit": f.unit,
               "readiness": f.readiness, "recent": f.recent, "version": f.version,
               "validation": validation(f, log, pack["fields"])} for f in pack["fields"]]
    if field_key is None:
        return {"version": pack["version"], "fields": fields}
    field = get_field(pack, field_key)
    if field is None:
        return {"status": "error", "reason": f"no field '{field_key}' in the pack"}
    extractions = [{"when": e["ts"], "report_id": e["report_id"], "version": e.get("field_version", 1),
                    "predicted": e["predicted"]["value"], "status": e["predicted"]["status"],
                    "confirmed": (e["confirmed"] or {}).get("value"), "outcome": e["outcome"],
                    "blind": e.get("blind", False), "readiness_after": e.get("readiness_after")}
                   for e in reversed(log) if e["field"] == field_key and e.get("phase") != "reset"]
    return {"field": {**next(f for f in fields if f["key"] == field_key), "meaning": field.meaning,
                      "definition": field.definition, "description": field.description, "range": field.range},
            "definition_history": field.history, "extraction_history": extractions}
