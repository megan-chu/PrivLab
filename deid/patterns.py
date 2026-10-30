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
from deid.ner import find_names
from deid.normalise import fold
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
# OCR tolerant: "0ng" (O read as 0), "L1m" (i read as 1) still count as name words.
WORD = rf"{NOT_NAME}(?:[A-Z]|0(?=[a-z]))(?:[a-z]|[01](?=[a-z])|'[A-Z])[A-Za-z'\-]*"
PARTICLE = r"(?:binte|binti|bin|bte|s/o|d/o|a/l|a/p|van|von|de|da)\b"   # longest first
SEP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"                       # spaces, or one line break
INITIALS = r"(?:[A-Z]\.[ \t]*)*"                          # "A. ", "L. M. "
NAME = rf"{WORD}(?:{SEP}(?:{PARTICLE}|{WORD}))+"         # two or more words
ONE_OR_MORE = rf"{INITIALS}{WORD}(?:{SEP}(?:{PARTICLE}|{WORD}))*"
AFTER_TITLE = r"[ \t]*\n?[ \t]*"                           # "Dr Koh", "Dr\nKoh"
DR_NAME = rf"Dr\.?{AFTER_TITLE}{ONE_OR_MORE}"             # "Dr" + one or more words
NOT_CAPS = (r"(?!(?:GROSS|DESCRIPTION|FINAL|DIAGNOSIS|MICROSCOPIC|CLINICAL|HISTORY|SPECIMEN|COMMENT|"
            r"REPORT|FROZEN|SECTION|PATHOLOGY|NOTE|SEE|AND|THE|OF)\b)")
CAPS_NAME = rf"{NOT_CAPS}[A-Z][A-Z'\-]+(?:[ \t]+{NOT_CAPS}[A-Z][A-Z'\-]+){{0,3}}"   # "TAN AH KOW"
LSEP = r"[ \t]*(?:\([^)\n]{0,20}\))?[ \t:#.\-]*"           # label -> value: "DOB (dd/mm/yyyy): ", "MRN-"
D = r"[\dOl]"                                               # a digit, allowing OCR O for 0 and l for 1
TITLES = ("Mr", "Mrs", "Ms", "Mdm", "Madam", "Miss", "Sister")
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
RELATIONS = (r"(?:son|daughter|wife|husband|brother|sister|mother|father|partner|niece|nephew|"
             r"friend|carer|caregiver|guardian|relative|cousin|aunt|uncle)")
HOSPITAL = (r"(?:St\.? )?(?:[A-Z][A-Za-z'\-]+ ){1,5}(?:General Hospital|Hospital|Cancer Institute|"
            rf"Cancer Cent(?:er|re)|Medical Cent(?:er|re)|Infirmary)")
MRN_VALUE = r"[A-Z]{0,3}\d{6,8}"
ORDINAL = r"(?:st|nd|rd|th)?"
DATE_ANY = (rf"(?:{D}{{1,2}}[/.\-]{D}{{1,2}}[/.\-](?:{D}{{4}}|{D}{{2}})|\d{{4}}-\d{{2}}-\d{{2}}|\d{{8}}|"
            rf"\d{{1,2}}{ORDINAL}[ \-]{MONTHS}[ \-,]*(?:\d{{4}}|\d{{2}})|{MONTHS} \d{{1,2}}{ORDINAL},? \d{{4}}|"
            rf"{MONTHS} \d{{4}})")
STREET = r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Close|Lane|Ln|Way|Crescent|Cres)"   # longest first
MRN_OCR = r"[A-Z]{0,3}[\dO]{6,9}"
# Any report type (forms, claims, banking): a label that names an identifier, and a role label before a name.
ID_LABEL = (r"(?:[A-Za-z]+[ \t]+){0,3}(?i:passport|TIN|tax identification|IBAN|policy|licen[cs]e|plate|registration|"
            r"no\.?|number|ID|ref(?:erence)?(?:[ \t]+no\.?)?)(?![A-Za-z])")   # "Staff ID", "Passport No", "Case File Ref"
ID_TOKEN = r"[A-Z0-9](?:[A-Z0-9\-/.]*[A-Z0-9])?"
ID_VALUE = rf"{ID_TOKEN}(?:[ \t]{ID_TOKEN}){{0,3}}"                        # no end dot; 3+ digits (analyze)
CSEP = r"[ \t]*(?:\([^)\n]{0,20}\))?[ \t]*[:#][ \t]*"                    # "(TIN): ", " # "
NAME_LINE = rf"{WORD}(?:[ \t]+(?:{PARTICLE}|{WORD}))+"                         # a name on one line
ROLE = (r"(?i:name|owner|manager|holder|counsel|officer|investigator|insured|claimant|driver|applicant|customer|"
        r"client|beneficiary|witness|agent|representative|director|signatory|guardian|employee|account holder)")
PHONE_CUE = r"(?i:tel(?:ephone)?|phone|call|hp|h/p|mobile|mob|office|contact(?: no\.?)?|fax|line)"
PHONE_VALUE = r"(?=(?:[^\dOlI\n]{0,2}[\dOlI]){7})\(?\+?\d[\dOlI \t().\-]{5,16}[\dOlI]"


# --------------------------------------------------------------------------
# Recognisers
# --------------------------------------------------------------------------
class NricRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(supported_entity="ID", name="sg_nric", global_regex_flags=FLAGS,
                         patterns=[Pattern("nric", r"(?i:(?<![0-9])[STFGM][ .\-]{0,2}\d{3,4}[ .\-]?\d{3,4}"
                                                   r"[ .\-]{0,2}[A-Z](?![A-Za-z0-9]))", 0.5)])

    def validate_result(self, pattern_text: str) -> bool:
        return nric_valid(re.sub(r"[ .\-]", "", pattern_text))


class NhsRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(supported_entity="ID", name="uk_nhs", global_regex_flags=FLAGS,
                         patterns=[Pattern("nhs", r"\b\d{3}[ .\-]?\d{3}[ .\-]?\d{4}\b", 0.5)])

    def validate_result(self, pattern_text: str) -> bool:
        return nhs_valid(pattern_text.replace(".", ""))


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
                                rf"(?P<v>{ONE_OR_MORE}|{CAPS_NAME})"),
        ("NAME", "name_label", rf"(?:^|\|)[ \t]*(?i:name|patient name)[ \t]*[:|][ \t]*(?P<v>{NAME}|{CAPS_NAME})"),
        ("NAME", "relation_before", rf"\b{RELATIONS}(?:-in-law)?,?[ \t]+(?P<v>{ONE_OR_MORE})"),
        ("NAME", "relation_after", rf"(?P<v>{NAME})[ \t]*\({RELATIONS}(?:-in-law)?\)"),
        ("NAME", "footer", rf"^Page \d+ of \d+ *\| *(?P<v>{NAME}) *\|"),
        ("DOCTOR", "dr", rf"\b(?P<v>{DR_NAME})"),
        ("DOCTOR", "prof", rf"\bProf(?:essor)?\.?{AFTER_TITLE}(?P<v>{ONE_OR_MORE})"),
        ("NAME", "title", rf"\b(?:{'|'.join(TITLES)})\.?{AFTER_TITLE}(?P<v>{ONE_OR_MORE})"),
        ("DOCTOR", "dr_caps", rf"\b(?P<v>(?i:dr|prof)\.?[ \t]+{CAPS_NAME})\b"),
        ("NAME", "title_caps", rf"\b(?i:mr|mrs|ms|mdm|madam|miss)\.?[ \t]+(?P<v>{CAPS_NAME})\b"),
        ("ADDRESS", "address", r"^Address:[ \t]*(?P<v>[^\n]*[^\s])"),
        ("HOSPITAL", "hospital", rf"\b(?P<v>{HOSPITAL})\b"),
        # DOB in any format: dates are kept by default, so an unlabelled DOB would stay in the text.
        ("DOB", "dob", rf"\b(?i:DOB|D\.O\.B\.?|Date of birth|Birth ?date|Born|b\.){LSEP}(?:on[ \t]+)?"
                       rf"(?P<v>{DATE_ANY})"),
        ("ID", "nric_label", r"\b(?i:NRIC|FIN|UIN|IC(?: no\.?)?|Patient ID)(?:[ \t]*[:.])?[ \t]*"
                             r"(?P<v>[STFGMstfgm5][ .\-]{0,2}[\dOoIlS]{3,4}[ .\-]?[\dOoIlS]{3,4}[ .\-]{0,2}[A-Za-z0-9])"
                             r"(?![A-Za-z0-9])"),
        ("ID", "nhs_label", r"\b(?i:NHS)(?: (?i:no\.?|number))?[ \t:#.\-]*(?P<v>\d(?:[ .\-]?\d){9})(?!\d)"),
        ("ID", "ssn_label", r"\b(?i:SSN|Social security(?: number| no\.?)?)[ \t:#.\-]*(?P<v>\d{3}[ .\-]?\d{2}[ .\-]?\d{4})(?!\d)"),
        ("MRN", "mrn_label", rf"\b(?i:MRN|M\.R\.N\.?|Medical record(?: number| no\.?)?|Hosp(?:ital)? (?:no\.?|number)|"
                             rf"Patient no\.?|Case no\.?|Med\.? ?Rec(?:ord)?\.?,?(?: no\.?)?){LSEP}(?P<v>{MRN_OCR})(?![\dA-Za-z])"),
        ("ACCESSION", "lab_label", r"\b(?i:Lab(?:oratory)? no\.?|Accession(?: no\.?| number)?|Our ref|Ref(?:erence)?(?: no\.?)?)"
                                  r"[ \t:#.\-]*(?P<v>[A-Z]{1,3}[ \-/]?\d{2}[ \-/]?\d{4,6})\b"),
        ("PHONE", "phone_cue", rf"\b{PHONE_CUE}[ \t:#.]*(?P<v>{PHONE_VALUE})(?!\d)"),
        ("ADDRESS", "postal", r"\b(?i:postal code|post ?code|zip(?: code)?)(?:[ \t]*[:.])?[ \t]*"
                              r"(?P<v>\d{5,6}|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2})\b"),
        ("MRN", "footer_mrn", rf"^Page \d+ of \d+ *\|[^|\n]*\| *(?P<v>{MRN_VALUE}) *$"),
        ("ACCOUNT", "account", rf"\b(?i:ACCT|Account(?: no\.?| number)?){LSEP}(?P<v>\d{{5,}})"),
        ("ACCOUNT", "account_any", rf"\b(?i:ACCT|Account|IBAN)(?:[ \t]+(?i:no\.?|number))?{CSEP}(?P<v>{ID_VALUE})(?![\w-])"),
        ("ID", "id_label", rf"(?<![A-Za-z]){ID_LABEL}{CSEP}(?P<v>{ID_VALUE})(?![\w-])"),
        ("NAME", "role_label", rf"(?:^|[|;])[ \t]*(?:[A-Za-z]+[ \t]+){{0,3}}{ROLE}[ \t]*(?:\([^)\n]{{0,20}}\))?[ \t]*:"
                               rf"[ \t]*(?P<v>{NAME_LINE}|{CAPS_NAME})"),
        ("ADDRESS", "address_any", r"^[ \t]*(?:[A-Za-z]+[ \t]+){1,2}(?i:address):[ \t]*(?P<v>[^\n]*[^\s])"),
    ]

    def __init__(self):
        self.rules = [(t, name, re.compile(rx, FLAGS)) for t, name, rx in self.RULES]
        super().__init__(supported_entities=sorted({t for t, _, _ in self.RULES}), name="labelled")

    def load(self) -> None:
        pass

    # ALL-CAPS names: TCGA text is often in capitals ("DR. REPORTED BY"), so a capitalised run made
    # only of words from the report vocabulary is not a name.
    CAPS_RULES = {"dr_caps", "title_caps", "next_of_kin", "name_label", "role_label"}
    DIGIT_RULES = {"id_label", "account_any"}

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        from deid.hold import vocabulary

        out = []
        for entity, rule, rx in self.rules:
            for m in rx.finditer(text):
                value = re.sub(r"^(?i:dr|prof)\.?\s+", "", m.group("v"))
                if rule in self.DIGIT_RULES and sum(ch.isdigit() for ch in value) < 3:
                    continue                              # "NODES EXAMINED", "AGE/SX": not an identifier
                if rule in self.CAPS_RULES and value.isupper() and all(
                        w.lower() in vocabulary() for w in re.findall(r"[A-Za-z']+", value)):
                    continue
                out.append(RecognizerResult(entity, m.start("v"), m.end("v"), 0.85,
                                            recognition_metadata={"rule": f"labelled_{rule}"}))
        return out


RECOGNISERS = [
    NricRecognizer(),
    NhsRecognizer(),
    simple("ssn_format", "ID", 0.6, r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
    simple("email", "EMAIL", 0.9, r"\b[\w.+-]+ ?@ ?[\w-]+(?:[.,][\w-]+)+\b",
           r"\b[\w.+-]+ ?(?:\(at\)|\[at\]) ?[\w-]+(?:\.[\w-]+)+\b",       # "name (at) domain"
           r"\b[\w.+-]+ (?:at|\(at\)|\[at\]) [\w-]+(?: (?:dot|\.) [\w-]+)+\b"),   # "name at domain dot com"
    simple("phone", "PHONE", 0.7,
           r"\+\d{1,3}(?:[ -]?\d{2,5}){2,4}\b",
           r"\(\d{3}\) ?\d{3}-\d{4}\b",
           r"(?<![\d+\-])\b[689]\d{3}[ -]?\d{4}\b(?![\-\d])",   # Singapore local, no +65
           r"\(\+\d{1,3}\) ?\d{3,4}[ -]?\d{4}\b",                  # (+65) 9123 4567
           r"\b07\d{3} ?\d{6}\b",                                   # UK mobile
           r"\b\d{3}-\d{3}-\d{4}\b"),                               # US 555-231-8846
    simple("date", "DATE", 0.6,
           r"\b\d{1,2}/\d{1,2}/\d{4}\b",
           r"\b\d{4}-\d{2}-\d{2}\b",
           rf"\b\d{{1,2}} {MONTHS} \d{{4}}\b",
           rf"\b{MONTHS} \d{{1,2}}{ORDINAL},? \d{{4}}\b",
           r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
    simple("accession", "ACCESSION", 0.8, r"\b(?:PA|SP|AP|S)[ \-]?\d{2}[ \-/]?\d{5}\b"),
    simple("street", "ADDRESS", 0.75,
           rf"\b(?:Blk|Block)\.? ?\d+[A-Z]? (?:[A-Z][a-z]+ ){{1,4}}{STREET}\.?(?: \d+)?(?:,? ?#\d{{2}}-\d{{1,4}})?",
           rf"\b\d{{1,5}} (?:[A-Z][a-z]+ ){{1,3}}{STREET}\b",
           r"\b\d{1,5} (?:Jalan|Jln|Lorong|Lor)\.? [A-Z][a-z]+(?: [A-Z][a-z]+){0,2}",
           r"\bSingapore \d{6}\b",
           r"\bS\(?\d{6}\)?(?!\d)",                                  # SG postal code S520088
           r"\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b"),                   # UK postcode NE1 4XY
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


def leftover_identifiers(text: str, kept_types=(), allowed=(), tags=None) -> list[PhiSpan]:
    """Identifiers the patterns still find in tagged text (the leak check and output scan).

    Ignores types the profile keeps, user-confirmed false alarms, and spans that are only tags
    (e.g. "Address: [ADDRESS-1]"). With tags (PrivLab's own TagSpans), only real tags count;
    a look-alike tag typed into the report does not.
    """
    allow = {a.lower() for a in allowed}
    if tags is None:
        only_tags = lambda h: bool(TAG_ONLY.fullmatch(h.text))
    else:
        tag_chars = {i for t in tags for i in range(t.start, t.end)}
        only_tags = lambda h: all(i in tag_chars or text[i] in " \t\n,;" for i in range(h.start, h.end))
    return [h for h in find_pattern_phi(text, names_model=False)       # rules only: a model guess never blocks
            if h.type not in kept_types and h.text.lower() not in allow and not only_tags(h)]


def learned_recognisers(terms) -> list[PatternRecognizer]:
    """Presidio deny-list recognisers from the profile's always_remove list ({"text", "type"}).

    Single-word names ("Tan") are not reused across reports: matched in any case they also hit
    ordinary words ("tan-coloured"). Within a report, repeat mentions handle them.
    """
    by_type = {}
    for t in terms:
        if t["type"] in ("NAME", "DOCTOR") and len(t["text"].split()) < 2:
            continue
        by_type.setdefault(t["type"], []).append(t["text"])
    return [PatternRecognizer(supported_entity=t, name="learned", deny_list=words, deny_list_score=0.95,
                              global_regex_flags=re.MULTILINE | re.IGNORECASE)
            for t, words in by_type.items()]


def find_pattern_phi(text: str, report_id: str | None = None, learned=(), names_model: bool = True) -> list[PhiSpan]:
    """learned: the profile's always_remove entries, matched as whole words, any case.
    names_model: also run the local name model (deid/ner.py) to fill gaps; off for the leftover scans.

    Scans a folded copy (look-alike characters made plain, same length); spans keep the original text.
    """
    with timed("presidio", report_id=report_id, chars=len(text)):
        scan = fold(text)
        results = []
        for rec in RECOGNISERS + learned_recognisers(learned):
            for r in rec.analyze(scan, rec.supported_entities, nlp_artifacts=None):
                rule = (r.recognition_metadata or {}).get("rule") or rec.name
                results.append((f"pattern:{rule}", r))
        kept = resolve_overlaps(results)
        found = [(r.entity_type, r.start, r.end, source, round(r.score, 2)) for source, r in kept]
        with timed("ner", report_id=report_id, chars=len(text)):    # the model only fills gaps
            found += [] if not names_model else [("NAME", a, b, "ner:spacy", 0.6) for a, b in find_names(scan)
                      if all(b <= s or a >= e for _, s, e, _, _ in found)]
        found += propagate_names(scan, found)
    return [
        PhiSpan(id=f"p{i}", type=t, start=s, end=e, text=text[s:e], source=source, score=score)
        for i, (t, s, e, source, score) in enumerate(sorted(found, key=lambda x: x[1]), 1)
    ]


NAME_PART = re.compile(r"[A-Za-z][A-Za-z'\-]+")
NOT_A_PART = set(TITLES) | {"Dr", "Prof", "Professor", "bin", "binte", "binti", "bte", "van", "von", "de", "da"}


def propagate_names(text: str, found: list[tuple]) -> list[tuple]:
    """Same report: once a name is found, its other mentions ("Tan" after "Tan Ah Kow") are too.

    found: (type, start, end, source, score). A full multi-word name also matches in any case
    ("ong wei ling", "TAN AH KOW"); single parts match as whole, case-sensitive words, unless the
    part is also an ordinary report word ("Low", "Long"), which would mask "Low grade".
    """
    taken = [(s, e) for _, s, e, _, _ in found]
    parts, full_names = {}, {}
    for t, s, e, _, _ in found:
        if t in ("NAME", "DOCTOR"):
            words = [p for p in NAME_PART.findall(text[s:e]) if p not in NOT_A_PART]
            if len(words) >= 2:
                full_names.setdefault(" ".join(words).lower(), (words, t))
            for part in words:
                parts.setdefault(part, t)
    extra = []
    for words, t in full_names.values():
        pattern = r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, words)) + r"(?![A-Za-z])"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not any(a <= m.start() < b or a < m.end() <= b for a, b in taken):
                extra.append((t, m.start(), m.end(), "pattern:repeat", 0.8))
                taken.append((m.start(), m.end()))
    from deid.hold import vocabulary

    vocab = vocabulary()
    for part, t in parts.items():
        if part.lower() in vocab:          # "Low" (surname) vs "Low grade": leave lone words to later layers
            continue
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
