"""
sections.py - which report section a position falls in, found by code from the headings
(never taken from the model). Used by the section check and for ranking.

Priority: addendum > synoptic = diagnosis > microscopic > other > gross > history.
"""
import re

HEADING = re.compile(
    r"\b(ADDENDUM|AMENDED|CASE SYNOPSIS|SYNOPTIC(?: DATA)?|FINAL DIAGNOSIS|"
    r"PRE-?OP(?:ERATIVE)? DIAGNOSIS|POST-?OP(?:ERATIVE)? DIAGNOSIS|FROZEN SECTION DIAGNOSIS|"
    r"DIAGNOSIS|MICROSCOPIC(?: DESCRIPTION)?|GROSS(?: DESCRIPTION)?|"
    r"CLINICAL (?:DIAGNOSIS|HISTORY)|PATIENT HISTORY|HISTORY|COMMENT)\s*[:.]", re.I)
# "CLINICAL DATA:" is left out on purpose: in TCGA synoptic blocks it labels the PSA line inside the
# synoptic section, so it must not switch the section to history (PP0185, 29 Oct).
PRIORITY = {"addendum": 5, "synoptic": 4, "diagnosis": 4, "microscopic": 3, "other": 2, "comment": 2,
            "gross": 1, "history": 0}
WEAK = {"gross", "history"}           # a value only from here is never green


def _name(heading: str) -> str:
    h = heading.upper()
    if "ADDENDUM" in h or "AMENDED" in h:
        return "addendum"
    if "SYNOP" in h:
        return "synoptic"
    if any(k in h for k in ("PRE", "POST", "CLINICAL", "HISTORY", "FROZEN")):
        return "history"
    if "DIAGNOSIS" in h:
        return "diagnosis"
    if "MICROSCOPIC" in h:
        return "microscopic"
    if "GROSS" in h:
        return "gross"
    return "comment"


def section_at(text: str, pos: int) -> str:
    name = "other"
    for m in HEADING.finditer(text, 0, pos + 1):
        if m.start() > pos:
            break
        name = _name(m.group(1))
    return name
