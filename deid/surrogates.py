"""
surrogates.py - consistent tags: the same entity gets the same [TYPE-n] within a report.

Names: the patient (found after "Patient:" or in the footer) is [PATIENT-n]; other names
are [PERSON-n]; a shorter mention ("Tan", "Mr Tan") joins the full name it is part of.
National IDs are tagged by kind ([NRIC-n], [NHS-n], [SSN-n]). Numbers follow first appearance.
"""
import re

PATIENT_SOURCES = {"pattern:labelled_patient", "pattern:labelled_footer"}
ID_KIND = {"pattern:sg_nric": "NRIC", "pattern:uk_nhs": "NHS", "pattern:ssn_format": "SSN"}
NOT_PARTS = {"dr", "prof", "professor", "mr", "mrs", "ms", "mdm", "madam", "miss", "sister",
             "bin", "binte", "binti", "bte", "van", "von", "de", "da"}


def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def name_parts(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z][a-z'\-]+", norm(text)) if w not in NOT_PARTS)


def assign_tags(spans) -> dict[str, str]:
    """span.id -> tag, for non-overlapping PhiSpans of one report."""
    patient_parts = frozenset().union(*[name_parts(s.text) for s in spans
                                        if s.type == "NAME" and s.source in PATIENT_SOURCES])

    def category(s) -> str:
        if s.type == "NAME":
            parts = name_parts(s.text)
            is_patient = s.source in PATIENT_SOURCES or (parts and parts <= patient_parts)
            return "PATIENT" if is_patient else "PERSON"
        if s.type == "ID":
            return ID_KIND.get(s.source, "ID")
        return s.type

    group = {"PATIENT": "NAME", "PERSON": "NAME"}
    entities = []   # dicts: category, key, parts, first, ids
    # Longer names first, so "Tan Ah Kow" exists before "Tan" looks for a match.
    for s in sorted(spans, key=lambda s: (-len(name_parts(s.text)), s.start)):
        cat, key, parts = category(s), norm(s.text), name_parts(s.text)
        match = next((e for e in entities if group.get(e["category"], e["category"]) == group.get(cat, cat)
                      and (e["key"] == key or (s.type in ("NAME", "DOCTOR") and parts and parts <= e["parts"]))),
                     None)
        if match:
            match["ids"].append(s.id)
            match["first"] = min(match["first"], s.start)
        else:
            entities.append({"category": cat, "key": key, "parts": parts, "first": s.start, "ids": [s.id]})

    tags, counts = {}, {}
    for e in sorted(entities, key=lambda e: e["first"]):
        counts[e["category"]] = counts.get(e["category"], 0) + 1
        for span_id in e["ids"]:
            tags[span_id] = f"[{e['category']}-{counts[e['category']]}]"
    return tags
