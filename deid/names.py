"""
names.py - Layer 2: Muse Spark double-checks confirmed, masked text for identifiers
the patterns missed (uncued names, places, organisations).

The model only proposes strings; code finds them in the text, skips tags and maps
offsets back to the original. A proposed string not found in the text is dropped.
"""
import re
from typing import Literal

from pydantic import BaseModel

from common import spark
from deid.mask import inside_tag, to_original
from deid.types import MaskedText, PhiSpan

INSTRUCTIONS = (
    "This is a pathology report. Identifiers already removed are shown as tags in square brackets, "
    "e.g. [PATIENT-1]. List every remaining identifier: names of people (patients, relatives, staff), "
    "places, addresses, organisations, phone numbers and ID numbers. Copy each one exactly as written. "
    "Do not list tags, clinical dates, medical terms or eponyms (e.g. Gleason, Nottingham, "
    "Bloom-Richardson), measurements or ordinary words. If there are none, return an empty list."
)
KIND_TO_TYPE = {"PERSON": "NAME", "PLACE": "ADDRESS", "ORGANISATION": "HOSPITAL",
                "PHONE": "PHONE", "ID_NUMBER": "ID"}


class Finding(BaseModel):
    text: str
    kind: Literal["PERSON", "PLACE", "ORGANISATION", "PHONE", "ID_NUMBER"]


class Findings(BaseModel):
    findings: list[Finding]


def locate(text: str, needle: str) -> list[tuple[int, int]]:
    """Whole-word occurrences; whitespace-flexible; exact case first, then any case."""
    words = needle.split()
    if not words:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    for flags in (0, re.IGNORECASE):
        hits = [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]
        if hits:
            return hits
    return []


def find_names(masked: MaskedText, original: str, report_id: str | None = None) -> tuple[list[PhiSpan], int]:
    """Returns (new spans in original offsets, number of proposals not found in the text)."""
    reply = spark.call(masked, instructions=INSTRUCTIONS, schema=Findings, step="names", report_id=report_id)
    allowed = {a.lower() for a in masked.allowed}
    spans, not_found = [], 0
    for f in reply.findings:
        if f.text.lower() in allowed:
            continue
        hits = [(s, e) for s, e in locate(masked.text, f.text) if not inside_tag(masked.tags, s, e)]
        if not hits:
            not_found += 1
            continue
        for s, e in hits:
            o_start, o_end = to_original(masked.tags, s, e)
            spans.append(PhiSpan(id=f"s{len(spans) + 1}", type=KIND_TO_TYPE[f.kind], start=o_start, end=o_end,
                                 text=original[o_start:o_end], source="spark"))
    return spans, not_found
