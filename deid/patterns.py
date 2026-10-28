"""
patterns.py - Layer 1: find structured identifiers with Presidio pattern recognisers.

Runs on the laptop. No spaCy model and no NER: each recogniser is called
directly with nlp_artifacts=None. Recognisers are built once, at import.

  find_pattern_phi(text) -> list[PhiSpan]    offsets into the original text

Patterns were shaped on dev reports only; extra reports check they generalise.

Usage (one report, compared with the answer key; dev/extra only):
  python -m deid.patterns PP0144
"""
import re
import sys

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

from common.timing import timed
from deid.checksums import nhs_valid, nric_valid
from deid.types import PhiSpan

# Presidio's default flags include IGNORECASE; IDs and labels here are case-sensitive.
FLAGS = re.MULTILINE

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
# A name word is Title-case (not ALL CAPS, which in TCGA text is a section heading) and not a common word.
NOT_NAME = (r"(?!(?:In|The|This|That|Of|And|For|To|On|At|By|With|Surgical|Pathology|Pathologist|"
            r"Department|Microscopic|Microscopy|Light|Gross|Final|Clinical|History|Comment|Report|"
            r"Frozen|Section|Attending|Specimen|Diagnosis)\b)")
WORD = rf"{NOT_NAME}[A-Z](?:[a-z]|'[A-Z])[A-Za-z'\-]*"
PARTICLE = r"(?:binte|binti|bin|bte|s/o|d/o|a/l|a/p|van|von|de|da)\b"   # longest first
SEP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"                       # spaces, or one line break
INITIALS = r"(?:[A-Z]\.[ \t]*)*"                          # "A. ", "L. M. "
NAME = rf"{WORD}(?:{SEP}(?:{PARTICLE}|{WORD}))+"         # two or more words
ONE_OR_MORE = rf"{INITIALS}{WORD}(?:{SEP}(?:{PARTICLE}|{WORD}))*"
DR_NAME = rf"Dr\.?[ \t]*{ONE_OR_MORE}"                    # "Dr" + one or more words
TITLES = ("Mr", "Mrs", "Ms", "Mdm", "Madam", "Miss", "Sister")
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
RELATIONS = (r"(?:son|daughter|wife|husband|brother|sister|mother|father|partner|niece|nephew|"
             r"friend|carer|caregiver|guardian|relative|cousin|aunt|uncle)")
HOSPITAL = (r"(?:St\.? )?(?:[A-Z][A-Za-z'\-]+ ){1,5}(?:General Hospital|Hospital|Cancer Institute|"
            rf"Cancer Cent(?:er|re)|Medical Cent(?:er|re)|Infirmary)")
MRN_VALUE = r"[A-Z]{0,3}\d{6,8}"


# --------------------------------------------------------------------------
# Recognisers
# --------------------------------------------------------------------------
class NricRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(supported_entity="ID", name="sg_nric", global_regex_flags=FLAGS,
                         patterns=[Pattern("nric", r"\b[STFGM]\d{7}[A-Z]\b", 0.5)])

    def validate_result(self, pattern_text: str) -> bool:
        return nric_valid(pattern_text)


class NhsRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(supported_entity="ID", name="uk_nhs", global_regex_flags=FLAGS,
                         patterns=[Pattern("nhs", r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b", 0.5)])

    def validate_result(self, pattern_text: str) -> bool:
        return nhs_valid(pattern_text)


def simple(name: str, entity: str, score: float, *regexes: str) -> PatternRecognizer:
    return PatternRecognizer(supported_entity=entity, name=name, global_regex_flags=FLAGS,
                             patterns=[Pattern(f"{name}_{i}", rx, score) for i, rx in enumerate(regexes)])


class LabelledRecognizer(EntityRecognizer):
    """Values that follow a label or cue, e.g. 'Patient: <name>'. The span covers the value only."""

    RULES = [
        ("NAME", "patient", rf"^Patient:[ \t]*(?P<v>{NAME})"),
        ("NAME", "relative", rf"\b[Pp]atient's {RELATIONS} (?P<v>{NAME})"),
        ("NAME", "request_of", rf"\bat the request of (?P<v>{NAME})"),
        ("NAME", "next_of_kin", rf"\b(?:Next of kin|NOK|Emergency contact)\b:?[ \t]*(?:{RELATIONS},?[ \t]+)?"
                                rf"(?P<v>{ONE_OR_MORE})"),
        ("NAME", "relation_before", rf"\b{RELATIONS}(?:-in-law)?,?[ \t]+(?P<v>{ONE_OR_MORE})"),
        ("NAME", "relation_after", rf"(?P<v>{NAME})[ \t]*\({RELATIONS}(?:-in-law)?\)"),
        ("NAME", "footer", rf"^Page \d+ of \d+ *\| *(?P<v>{NAME}) *\|"),
        ("DOCTOR", "dr", rf"\b(?P<v>{DR_NAME})"),
        ("DOCTOR", "prof", rf"\bProf(?:essor)?\.?[ \t]*(?P<v>{ONE_OR_MORE})"),
        ("NAME", "title", rf"\b(?:{'|'.join(TITLES)})\.?[ \t]*(?P<v>{ONE_OR_MORE})"),
        ("ADDRESS", "address", r"^Address:[ \t]*(?P<v>[^\n]*[^\s])"),
        ("HOSPITAL", "hospital", rf"\b(?P<v>{HOSPITAL})\b"),
        ("DOB", "dob", r"\bDOB:[ \t]*(?P<v>\d{1,2}/\d{1,2}/\d{4})"),
        ("MRN", "mrn_label", rf"\bMRN:[ \t]*(?P<v>{MRN_VALUE})\b"),
        ("MRN", "footer_mrn", rf"^Page \d+ of \d+ *\|[^|\n]*\| *(?P<v>{MRN_VALUE}) *$"),
        ("ACCOUNT", "account", r"\bACCT *#? *: *(?P<v>\d{5,})"),
    ]

    def __init__(self):
        self.rules = [(t, name, re.compile(rx, FLAGS)) for t, name, rx in self.RULES]
        super().__init__(supported_entities=sorted({t for t, _, _ in self.RULES}), name="labelled")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        out = []
        for entity, rule, rx in self.rules:
            for m in rx.finditer(text):
                out.append(RecognizerResult(entity, m.start("v"), m.end("v"), 0.85,
                                            recognition_metadata={"rule": f"labelled_{rule}"}))
        return out


RECOGNISERS = [
    NricRecognizer(),
    NhsRecognizer(),
    simple("ssn_format", "ID", 0.6, r"\b\d{3}-\d{2}-\d{4}\b"),
    simple("email", "EMAIL", 0.9, r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    simple("phone", "PHONE", 0.7,
           r"\+\d{1,3}(?:[ -]?\d{2,5}){2,4}\b",
           r"\(\d{3}\) ?\d{3}-\d{4}\b",
           r"(?<![\d+\-])\b[689]\d{3}[ -]?\d{4}\b(?![\-\d])"),   # Singapore local, no +65
    simple("date", "DATE", 0.6,
           r"\b\d{1,2}/\d{1,2}/\d{4}\b",
           r"\b\d{4}-\d{2}-\d{2}\b",
           rf"\b\d{{1,2}} {MONTHS} \d{{4}}\b",
           rf"\b{MONTHS} \d{{1,2}},? \d{{4}}\b"),
    simple("accession", "ACCESSION", 0.8, r"\b(?:PA|SP|S)\d{2}-?\d{5}\b"),
    simple("mrn", "MRN", 0.9, r"\bMRN\d{6,8}\b"),
    LabelledRecognizer(),
]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def resolve_overlaps(results: list[tuple[str, RecognizerResult]]) -> list[tuple[str, RecognizerResult]]:
    """Keep non-overlapping spans: longer first, then higher score."""
    ordered = sorted(results, key=lambda x: (-(x[1].end - x[1].start), -x[1].score, x[1].start))
    kept = []
    for source, r in ordered:
        if all(r.end <= k.start or r.start >= k.end for _, k in kept):
            kept.append((source, r))
    return sorted(kept, key=lambda x: x[1].start)


TAG_ONLY = re.compile(r"(?:\s*\[[A-Z]+-\d+\]\s*[,;]?)+")


def leftover_identifiers(text: str, kept_types=(), allowed=()) -> list[PhiSpan]:
    """Identifiers the patterns still find in tagged text (the leak check and output scan).

    Ignores types the profile keeps, user-confirmed false alarms, and spans that are only tags
    (e.g. "Address: [ADDRESS-1]").
    """
    allow = {a.lower() for a in allowed}
    return [h for h in find_pattern_phi(text)
            if h.type not in kept_types and h.text.lower() not in allow and not TAG_ONLY.fullmatch(h.text)]


def learned_recognisers(terms) -> list[PatternRecognizer]:
    """Presidio deny-list recognisers from the profile's always_remove list ({"text", "type"})."""
    by_type = {}
    for t in terms:
        by_type.setdefault(t["type"], []).append(t["text"])
    return [PatternRecognizer(supported_entity=t, name="learned", deny_list=words, deny_list_score=0.95,
                              global_regex_flags=re.MULTILINE | re.IGNORECASE)
            for t, words in by_type.items()]


def find_pattern_phi(text: str, report_id: str | None = None, learned=()) -> list[PhiSpan]:
    """learned: the profile's always_remove entries, matched as whole words, any case."""
    with timed("presidio", report_id=report_id, chars=len(text)):
        results = []
        for rec in RECOGNISERS + learned_recognisers(learned):
            for r in rec.analyze(text, rec.supported_entities, nlp_artifacts=None):
                rule = (r.recognition_metadata or {}).get("rule") or rec.name
                results.append((f"pattern:{rule}", r))
        kept = resolve_overlaps(results)
        found = [(r.entity_type, r.start, r.end, source, round(r.score, 2)) for source, r in kept]
        found += propagate_names(text, found)
    return [
        PhiSpan(id=f"p{i}", type=t, start=s, end=e, text=text[s:e], source=source, score=score)
        for i, (t, s, e, source, score) in enumerate(sorted(found, key=lambda x: x[1]), 1)
    ]


NAME_PART = re.compile(r"[A-Za-z][A-Za-z'\-]+")
NOT_A_PART = set(TITLES) | {"Dr", "Prof", "Professor", "bin", "binte", "binti", "bte", "van", "von", "de", "da"}


def propagate_names(text: str, found: list[tuple]) -> list[tuple]:
    """Same report: once a name is found, its other mentions ("Tan" after "Tan Ah Kow") are too.

    found: (type, start, end, source, score). Matches each name part as a whole, case-sensitive word.
    """
    taken = [(s, e) for _, s, e, _, _ in found]
    parts = {}
    for t, s, e, _, _ in found:
        if t in ("NAME", "DOCTOR"):
            for part in NAME_PART.findall(text[s:e]):
                if part not in NOT_A_PART:
                    parts.setdefault(part, t)
    extra = []
    for part, t in parts.items():
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(part)}(?![A-Za-z])", text):
            if not any(a <= m.start() < b or a < m.end() <= b for a, b in taken):
                extra.append((t, m.start(), m.end(), "pattern:repeat", 0.8))
                taken.append((m.start(), m.end()))
    return extra


def main(report_id: str) -> int:
    from common.data import load_phi_truth, load_report

    truth = load_phi_truth(report_id)
    if truth["split"] == "test":
        print(f"{report_id} is in the test split; open it only for the final evaluation.")
        return 2
    text = load_report(report_id)
    spans = find_pattern_phi(text, report_id=report_id)

    print(f"{report_id} ({truth['split']}, {truth['country']}): {len(spans)} found, {len(truth['phi'])} in answer key\n")
    for s in spans:
        hit = any(t["start"] == s.start and t["end"] == s.end and t["type"] == s.type for t in truth["phi"])
        print(f"  {'ok ' if hit else 'FP?'} {s.type:9} {s.source:27} {s.start:>5}-{s.end:<5} {s.text}")
    missed = [t for t in truth["phi"] if not any(s.start <= t["start"] and s.end >= t["end"] for s in spans)]
    for t in missed:
        print(f"  MISS {t['type']:9} {'':27} {t['start']:>5}-{t['end']:<5} {t['text']}")
    print(f"\n{len(missed)} missed")
    return 1 if missed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m deid.patterns <report_id>")
    sys.exit(main(sys.argv[1]))
