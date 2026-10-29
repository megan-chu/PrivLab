"""
checks.py - the four in-product checks on every candidate (PLAN.md §4), all deterministic:

  evidence_found     the evidence text exists in the de-identified report (we find its position)
  type_ok            the value parses as the field's type and lies in its range / choices
  value_in_evidence  the value as written appears in its evidence (no invented numbers)
  section_ok         the evidence is not only from clinical history or the gross description

Plus an output scan: no identifier pattern may appear in a value or evidence quote.
"""
import re

from deid.names import locate
from deid.patterns import leftover_identifiers
from extract.sections import WEAK, section_at

NUMBER = re.compile(r"\d+(?:\.\d+)?")
TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0}


def parse_number(value_text: str, unit_text: str | None, field_unit: str | None) -> float | None:
    m = NUMBER.search(value_text.replace(",", "."))
    if not m:
        return None
    value = float(m.group())
    unit = (unit_text or field_unit or "").strip().lower()
    if field_unit and unit != field_unit.lower() and unit in TO_CM and field_unit.lower() in TO_CM:
        value = value * TO_CM[unit] / TO_CM[field_unit.lower()]
    return round(value, 3)


def run_checks(field, cand, text: str, kept_types=()) -> dict:
    """Returns {"checks": {...}, "value", "start", "end", "section", "fail"}; fail = first failed check."""
    out = {"checks": {}, "value": None, "start": None, "end": None, "section": None, "fail": None}
    hits = locate(text, cand.evidence_text)
    out["checks"]["evidence_found"] = bool(hits)
    if hits:
        out["start"], out["end"] = hits[0]
        out["section"] = section_at(text, hits[0][0])
    evidence = text[out["start"]:out["end"]] if hits else cand.evidence_text

    if field.type == "number":
        value = parse_number(cand.value_text, cand.unit_text, field.unit)
        lo, hi = field.range if field.range else (None, None)
        out["checks"]["type_ok"] = value is not None and (lo is None or value >= lo) and (hi is None or value <= hi)
        number = NUMBER.search(cand.value_text.replace(",", "."))
        out["checks"]["value_in_evidence"] = bool(number) and re.search(
            rf"(?<![\d.]){re.escape(number.group())}(?![\d])", evidence.replace(",", ".")) is not None
    else:
        choice = next((c for c in field.choices if c.lower() == cand.value_text.strip().lower()), None)
        value = choice
        out["checks"]["type_ok"] = choice is not None
        words = [w for w in re.findall(r"[a-z0-9]+", (choice or cand.value_text).lower()) if len(w) > 2]
        out["checks"]["value_in_evidence"] = bool(words) and all(w in evidence.lower() for w in words)
    out["value"] = value
    out["checks"]["section_ok"] = out["section"] not in WEAK if hits else False
    out["checks"]["no_identifiers"] = not leftover_identifiers(f"{cand.value_text}\n{evidence}", kept_types)

    for name in ("evidence_found", "no_identifiers", "type_ok", "value_in_evidence"):
        if not out["checks"][name]:
            out["fail"] = name
            break
    return out
