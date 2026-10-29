"""
names.py - Layer 2, the Muse Spark name check: a model double-checks confirmed, masked text for
identifiers the local rules missed (uncued names, places, organisations).

Built to survive a weaker or newly swapped model:
  - The local layer passes "suspects" (words it is unsure about) and the model must rule on each:
    an identifier (findings) or not (not_identifiers). Pointing a model at a word is more reliable
    than hoping it spots it.
  - A suspect the model says nothing about comes back as "unresolved", and the pipeline shows it to
    the user unticked: a silent model can no longer drop a name.
  - The model only proposes strings; code finds them in the text, skips tags and maps offsets back
    to the original. A proposed string not found in the text is dropped.

PROMPT_VERSION is part of a model's qualification (deid/qualify.py): change the prompt, re-qualify.
"""
import re
from typing import Literal

from pydantic import BaseModel

from common import spark
from deid.mask import inside_tag, to_original
from deid.types import MaskedText, PhiSpan

PROMPT_VERSION = "names-2"
INSTRUCTIONS = (
    "This is a pathology report. Identifiers already removed are shown as tags in square brackets, "
    "e.g. [PATIENT-1]. List every remaining identifier in findings: names of people (patients, relatives, "
    "staff), including a single first name or surname and names written in lower case or capitals; places, "
    "addresses, organisations, phone numbers and ID numbers. Copy each one exactly as written. "
    "Do not list tags, clinical dates, medical terms or eponyms (e.g. Gleason, Nottingham, "
    "Bloom-Richardson), measurements or ordinary words. If there are none, return an empty list."
)
SUSPECT_LINE = ("\nThese words may or may not be identifiers. Decide for EACH one: put it in findings if it "
                "is an identifier, or copy it into not_identifiers if it is not: ")
KIND_TO_TYPE = {"PERSON": "NAME", "PLACE": "ADDRESS", "ORGANISATION": "HOSPITAL",
                "PHONE": "PHONE", "ID_NUMBER": "ID"}


class Finding(BaseModel):
    text: str
    kind: Literal["PERSON", "PLACE", "ORGANISATION", "PHONE", "ID_NUMBER"]


class Findings(BaseModel):
    findings: list[Finding]
    not_identifiers: list[str]


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


def find_names(masked: MaskedText, original: str, report_id: str | None = None, *,
               suspects=(), provider: str | None = None) -> tuple[list[PhiSpan], int, list[PhiSpan]]:
    """suspects: (start, end) in the masked text. Returns (found spans, proposals not in the text,
    unresolved suspects), spans in original offsets."""
    words = sorted({masked.text[s:e] for s, e in suspects})
    instructions = INSTRUCTIONS + (SUSPECT_LINE + "; ".join(f'"{w}"' for w in words) if words else "")
    reply = spark.call(masked, instructions=instructions, schema=Findings, step="names",
                       report_id=report_id, provider=provider)
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

    ruled_out = {" ".join(w.split()).lower() for w in reply.not_identifiers}
    unresolved = []
    for s, e in suspects:
        o_start, o_end = to_original(masked.tags, s, e)
        covered = any(x.start <= o_start and o_end <= x.end for x in spans)
        if not covered and " ".join(masked.text[s:e].split()).lower() not in ruled_out:
            unresolved.append(PhiSpan(id=f"q{len(unresolved) + 1}", type="NAME", start=o_start, end=o_end,
                                      text=original[o_start:o_end], source="suspect"))
    return spans, not_found, unresolved
