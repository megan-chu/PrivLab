"""
candidates.py - find_candidates: likely values for a pack field in a de-identified report,
each with evidence, checks and a reason; best first; green, amber or missing.

One Muse Spark call per report covers every field in the pack (cached), and sees only the
sealed DeidResult text. The reply is frozen per report: later fields on the same report reuse it,
even after a correction changed the pack's examples, so teaching 7 fields costs one call per
report, not seven, and every field's prediction exists before any feedback on that report.
A field whose DEFINITION changed since (meaning, type, unit, range) gets a new call, so a
refined definition is previewed with its new meaning; learned examples alone never trigger one.
The model proposes; code finds the evidence, runs the checks, detects the section and decides
the status.

  find_candidates(deid_id, field_key) -> FieldResult dict
  prefetch(deid_id)                   -> makes the one call ahead of time (viewer, after de-id);
                                         no prediction is frozen and no review timer starts
  preview(deid_id)                    -> where each field's value sits, from that reply only (no call)
"""
import hashlib
import logging
import threading
import time

from common import spark
from common.config import ConfigError
from deid import pipeline
from extract.checks import norm_value, run_checks
from extract.pack import get_field, load_pack, meaning_key
from extract.sections import PRIORITY, WEAK
from extract.types import CandidatesReply

log = logging.getLogger("privlab.extract")

INSTRUCTIONS = (
    "You extract fields from a de-identified pathology report for a cancer registry. Identifiers are "
    "tags like [PATIENT-1]. For each field below, list every place the report states a value, best first. "
    "value_text and unit_text: exactly as written in the report. evidence_text: the shortest exact phrase "
    "that states the value (ideally under 15 words), copied character for character. reason: at most 10 words. "
    "confidence: 0 to 1. If a field is not stated, return an "
    "empty candidates list. Do not infer, calculate or give medical advice."
)
MIN_CONFIDENCE = 0.6
BLIND_EVERY = 10          # about 1 field result in 10 hides PrivLab's answer (viewer only)
_REPLIES: dict[str, tuple[CandidatesReply, int, dict]] = {}   # deid_id -> (reply, model ms, definitions)
_LOCKS: dict[str, threading.Lock] = {}                     # one call per report, even with a prefetch running
_LOCKS_GUARD = threading.Lock()


def _field_spec(f) -> str:
    spec = f"- key: {f.key}\n  meaning: {f.meaning}" + (f"\n  definition: {f.definition}" if f.definition else "")
    spec += f"\n  type: {f.type}"
    if f.type == "number":
        spec += f", unit {f.unit or '-'}" + (f", range {f.range[0]}-{f.range[1]}" if f.range else "")
    else:
        spec += ", text exactly as written in the report"
    if f.multi:
        spec += "\n  a list: one candidate per item the report states (e.g. one per biomarker line), all of them"
    for ex in f.examples.get("good", [])[-3:]:
        spec += f"\n  correct example: \"{ex['context']}\" -> {ex['value']}"
    for ex in f.examples.get("bad", [])[-3:]:
        spec += f"\n  wrong example (do not pick): \"{ex['context']}\" ({ex.get('reason', '')})"
    return spec


def _reply_for(result, pack, effort=None) -> CandidatesReply:
    fields = "\n".join(_field_spec(f) for f in pack["fields"])
    return spark.call(result, instructions=f"{INSTRUCTIONS}\n\nFields:\n{fields}", schema=CandidatesReply,
                      step="candidates", report_id=result.report_id, effort=effort)


def _frozen_reply(deid_id: str, result, pack, field_key: str, effort=None) -> tuple[CandidatesReply, int]:
    """The report's reply (cached per report); a new call only if none yet covers field_key under
    its current definition."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(deid_id, threading.Lock())
    with lock:
        frozen = _REPLIES.get(deid_id)
        current = {f.key: meaning_key(f) for f in pack["fields"]}
        if frozen and any(fc.key == field_key for fc in frozen[0].fields) \
                and frozen[2].get(field_key) == current.get(field_key):
            return frozen[0], 0                                # no new call, no wait
        t0 = time.perf_counter()
        reply = _reply_for(result, pack, effort)
        model_ms = round((time.perf_counter() - t0) * 1000)
        _REPLIES[deid_id] = (reply, model_ms, current)
        return reply, model_ms


def prefetch(deid_id: str, *, pack_path=None, effort: str | None = None) -> dict:
    result = pipeline.result_for(deid_id)
    pack = load_pack(pack_path)
    if result is None or not pack["fields"]:
        return {"deid_id": deid_id, "status": "skipped"}
    try:
        _, model_ms = _frozen_reply(deid_id, result, pack, pack["fields"][-1].key, effort)
    except Exception as e:   # the click later shows amber "unavailable"; never a crash
        return {"deid_id": deid_id, "status": "error", "reason": type(e).__name__}
    return {"deid_id": deid_id, "status": "ready", "model_ms": model_ms}


def forget_reply(deid_id: str) -> None:
    """Drop a report's frozen reply, so its next read uses what PrivLab has learned since ("apply to other
    reports"). The prompt holds the new examples, so that read is a new call; an unchanged prompt hits the cache."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(deid_id, threading.Lock())
    with lock:
        _REPLIES.pop(deid_id, None)


def preview(deid_id: str, *, pack_path=None) -> dict:
    """Where each field's value sits, from the read-ahead reply only: never a model call, no prediction
    frozen, no review timer. A field read under an older definition, or due a blind check, is left out."""
    result, frozen = pipeline.result_for(deid_id), _REPLIES.get(deid_id)
    if result is None or frozen is None:
        return {"deid_id": deid_id, "status": "none"}
    reply, _, defs = frozen
    out = {}
    for f in load_pack(pack_path)["fields"]:
        if defs.get(f.key) != meaning_key(f) or (not f.multi and _is_blind(deid_id, f.key)):
            continue
        fc = next((x for x in reply.fields if x.key == f.key), None)
        spots = []
        for c in fc.candidates if fc else []:
            checked = run_checks(f, c, result.text, result.kept_types)
            if checked["fail"] is None and checked["start"] is not None:
                spots.append({"start": checked["start"], "end": checked["end"], "value": checked["value"]})
        out[f.key] = spots if f.multi else spots[:1]
    return {"deid_id": deid_id, "status": "ready", "fields": out}


def find_candidates(deid_id: str, field_key: str, *, pack_path=None, effort: str | None = None,
                    blind_checks: bool = False, log_path=None) -> dict:
    """blind_checks: sometimes hide the answer, so the user's label is not anchored on it.
    log_path: the score log, to find the user's earlier answer on the same text (answer memory)."""
    blind = blind_checks and _is_blind(deid_id, field_key)
    return _find(deid_id, field_key, pack_path, effort, blind, log_path)   # a list field is never blind (see _find)


def _is_blind(deid_id: str, field_key: str) -> bool:
    return int(hashlib.sha256(f"{deid_id}:{field_key}".encode()).hexdigest(), 16) % BLIND_EVERY == 0


def _find(deid_id: str, field_key: str, pack_path, effort, blind: bool, log_path=None) -> dict:
    result = pipeline.result_for(deid_id)
    if result is None:
        return {"deid_id": deid_id, "status": "error", "reason": "de-identify and confirm this report first"}
    pack = load_pack(pack_path)
    field = get_field(pack, field_key)
    if field is None:
        return {"deid_id": deid_id, "status": "error", "reason": f"no field '{field_key}' in the pack; define it first"}
    blind = blind and not field.multi
    if not field.multi:                       # the same report again: the user's own answer, no model call
        from extract.learn import remembered
        mem = remembered(pipeline.fingerprint(deid_id), field, log_path)
        out = mem and _remembered(deid_id, field, mem, result.text)
        if out:
            return out

    try:
        reply, model_ms = _frozen_reply(deid_id, result, pack, field.key, effort)
    except Exception as e:   # API down, timeout, bad JSON: amber, never a crash
        why = (f"{spark.OFFLINE}: extraction needs Muse Spark. Try again when online, or enter the value"
               if isinstance(e, spark.OfflineError) else f"{e} Enter the value yourself for now"
               if isinstance(e, ConfigError) else f"Muse Spark unavailable ({type(e).__name__}); review manually")
        return _result(deid_id, field, "amber", None, [], why)

    proposed = next((fc.candidates for fc in reply.fields if fc.key == field.key), [])
    cands = []
    for i, c in enumerate(proposed, 1):
        checked = run_checks(field, c, result.text, result.kept_types)
        cands.append({"id": f"c{i}", "value": checked["value"], "unit": field.unit if field.type == "number" else None,
                      "evidence_text": c.evidence_text, "start": checked["start"], "end": checked["end"],
                      "section": checked["section"], "reason": c.reason, "confidence": round(c.confidence, 2),
                      "checks": checked["checks"], "fail": checked["fail"]})

    _apply_rules(field, cands)
    valid = [c for c in cands if c["fail"] is None]
    valid.sort(key=lambda c: (PRIORITY.get(c["section"], 2), c["confidence"]), reverse=True)
    ranked = valid + [c for c in cands if c["fail"] is not None]
    for rank, c in enumerate(ranked, 1):
        c["rank"] = rank

    if not cands:
        return _result(deid_id, field, "missing", None, ranked, "Not stated in this report", model_ms, blind)
    if field.multi:                           # every verified candidate is an item; one failed check makes it amber
        items = _items(valid)
        if not items:
            return _result(deid_id, field, "amber", None, ranked, f"No item passed the checks ({ranked[0]['fail']})", model_ms)
        failed = len(cands) - len(valid)
        reason = f"{failed} of {len(cands)} items could not be verified in the text" if failed else None
        return _result(deid_id, field, "amber" if failed else "green", None, ranked, reason, model_ms, items=items)
    if not valid:
        return _result(deid_id, field, "amber", None, ranked, f"No candidate passed the checks ({ranked[0]['fail']})", model_ms, blind)
    top = valid[0]
    second = None if blind else _second_opinion(result, field, valid)
    sure = second is not None and second["best"] is not None and second["p"] >= SURE
    if sure:
        top = second["best"]                                 # the second opinion picks among the candidates
    rivals = [c for c in valid if c is not top and norm_value(c["value"]) != norm_value(top["value"])
              and PRIORITY.get(c["section"], 2) == PRIORITY.get(top["section"], 2)]
    doubt = second is not None and second["of"].get(top["id"], 0) < DOUBT
    if top.get("rule") and not top["trusted"]:          # a rule's first use: the user says if it is right
        reason = f"From your correction: \"{top['raw']}\" means {top['value']}. Right?"
    elif rivals and not sure:
        reason = f"Two different values in the same section: {top['value']} vs {rivals[0]['value']}. Which one?"
    elif doubt:
        reason = f"A second check doubts this value ({second['of'].get(top['id'], 0):.0%} sure)"
    elif top["section"] in WEAK:
        reason = f"From the {top['section']} section only"
    elif top["confidence"] < MIN_CONFIDENCE:
        reason = "Low confidence"
    else:
        return _result(deid_id, field, "green", top, ranked, None, model_ms, blind, second=_shown(second, top))
    return _result(deid_id, field, "amber", top, ranked, reason, model_ms, blind, second=_shown(second, top))


SURE, DOUBT = 0.9, 0.5      # the second opinion picks a value at >= 90%; a value it gives < 50% is doubted


def _second_opinion(result, field, valid: list) -> dict | None:
    """Baseten's decision model reads the masked report and says which candidate it states (or none):
    {"best": candidate or None, "p": its probability, "of": candidate id -> probability}. None if unavailable."""
    options, seen = {}, set()
    for c in valid:
        if norm_value(c["value"]) not in seen and len(options) < 4:
            seen.add(norm_value(c["value"]))
            options[chr(97 + len(options))] = c
    try:
        answer = spark.decide(result, step="second", report_id=result.report_id,
                              context=f"Field: {field.name}. {field.definition or field.meaning}",
                              question={"type": "choice", "instructions": "Which value does this report state for "
                                        "the field, as the field is defined?",
                                        "criteria": {**{k: str(c["value"]) for k, c in options.items()},
                                                     "none": "None of these values"}})
    except Exception as e:      # down, slow, refused: extraction goes on exactly as without it
        log.warning("second opinion unavailable (%s)", type(e).__name__)
        return None
    if not answer:
        return None
    probs = answer.get("probabilities") or {}
    of = {}
    for k, c in options.items():            # one probability per value: a repeated value shares it
        for v in valid:
            if norm_value(v["value"]) == norm_value(c["value"]):
                of[v["id"]] = probs.get(k, 0.0)
    choice = answer.get("choice")
    return {"best": options.get(choice), "p": probs.get(choice, 0.0), "of": of}


def _shown(second: dict | None, top: dict) -> dict | None:
    """What the field card shows: how sure the second opinion is of the value PrivLab proposes."""
    return second and {"p": round(second["of"].get(top["id"], 0.0), 3),
                       "picked": second["best"]["value"] if second["best"] else None}


def _apply_rules(field, cands: list) -> None:
    """The user's corrections, learned as rules ("mammae sinistrae" -> left breast): code rewrites the model's
    value. The checks already ran on the words in the report, so the quote still proves where it came from."""
    if field.type != "text" or field.multi:
        return
    rules = {r["from"]: r for r in field.examples.get("rules", [])}
    for c in cands:
        rule = c["value"] is not None and rules.get(norm_value(c["value"]))
        if rule and norm_value(rule["to"]) != norm_value(c["value"]):
            c.update(raw=c["value"], value=rule["to"], rule=rule["from"], trusted=rule.get("uses", 0) > 0)


def _remembered(deid_id: str, field, mem: dict, text: str) -> dict | None:
    """The earlier answer, found again in this masking (it can shift as the profile learns); None if it is gone."""
    from deid.names import locate

    c, when = mem.get("confirmed"), mem["ts"][:10]
    top = None
    if c and c.get("value") is not None:
        hits = locate(text, str(c["value"]))
        if not hits:
            return None                           # not in this text any more: ask the model as usual
        start, end = min(hits, key=lambda h: abs(h[0] - (c.get("start") or 0)))
        top = {"id": "c1", "value": c["value"], "unit": field.unit if field.type == "number" else None,
               "evidence_text": text[start:end], "start": start, "end": end, "section": None,
               "reason": "your earlier answer", "confidence": 1.0, "checks": {}, "fail": None, "rank": 1}
    return _result(deid_id, field, "remembered", top, [top] if top else [], f"Your answer from {when}", 0)


def _items(valid: list) -> list:
    """The verified candidates as list items, in report order, one per quote."""
    seen, items = set(), []
    for c in sorted(valid, key=lambda c: c["start"]):
        if (c["start"], c["end"]) not in seen:
            seen.add((c["start"], c["end"]))
            items.append({k: c[k] for k in ("id", "value", "start", "end", "evidence_text", "section")})
    return items


def _result(deid_id: str, field, status: str, top: dict | None, ranked: list, reason: str | None,
            model_ms: int | None = None, blind: bool = False, items: list | None = None,
            second: dict | None = None) -> dict:
    from extract.learn import remember_prediction

    result = {**_build(deid_id, field, status, top, ranked, reason), "model_ms": model_ms, "blind": blind,
              **({"second": second} if second else {})}
    if field.multi:
        result.update(items=items or [], value=[i["value"] for i in items or []])
    remember_prediction(result)                 # frozen before any feedback ("test then learn")
    if blind:                                   # the user answers without seeing it
        return {**_build(deid_id, field, "blind", None, [], "Blind check: find the value yourself"),
                "model_ms": model_ms, "blind": True}
    return result


def _build(deid_id: str, field, status: str, top: dict | None, ranked: list, reason: str | None) -> dict:
    return {"deid_id": deid_id, "field": field.key, "status": status,
            "value": top["value"] if top else None, "unit": top["unit"] if top else None,
            "candidate_id": top["id"] if top else None, "evidence_text": top["evidence_text"] if top else None,
            "start": top["start"] if top else None, "end": top["end"] if top else None,
            "section": top["section"] if top else None, "reason": reason,
            "candidates": ranked, "readiness": field.readiness}
