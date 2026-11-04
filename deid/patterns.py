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
from deid.checksums import iban_valid, luhn_valid, nhs_valid, nric_valid
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
WORD = rf"{NOT_NAME}(?:[A-Z]|0(?=[a-z]))(?:[a-z]|[01](?=[01]?[a-z])|'[A-Z])(?:[A-Za-z'\-]|[01](?=[a-z]))*"   # "B00n"
PARTICLE = r"(?:binte|binti|bin|bte|s/o|d/o|a/l|a/p|van|von|de|da)\b"   # longest first
# A name goes on to the next line only if that line does not start a label of up to four words ("Date Of Birth:",
# "Referring Doctor:") or a field without a colon ("Age 54", "Sex M").
NEXT_LINE_OK = (r"(?![A-Za-z][A-Za-z'\-]*(?:[ \t]+[A-Za-z][A-Za-z'\-]*){0,3}[ \t]*:)"
                r"(?!(?i:age|sex|gender|dob|date|mrn|nric|phone|tel|address|ward|bed|room|unit)\b)")
CONT = (rf"(?:[ \t]+(?:[A-Z]\.|(?![AI][ \t])[A-Z])(?=[ \t]+[A-Z][a-z])"   # middle initial: "Jane K. Zhao" (not "A", "I")
        rf"|[ \t]+(?:{PARTICLE}|{WORD})"                  # same line: any word
        rf"|[ \t]*\n[ \t]*{NEXT_LINE_OK}(?:{PARTICLE}|{WORD}))")   # next line: not a "Label:" nor a field
INITIALS = r"(?:[A-Z]\.[ \t]*)*"                          # "A. ", "L. M. "
NAME = rf"{WORD}(?:{CONT})+"                              # two or more words
ONE_OR_MORE = rf"{INITIALS}{WORD}(?:{CONT})*"
AFTER_TITLE = r"[ \t]*\n?[ \t]*"                           # "Dr Koh", "Dr\nKoh"
DR_NAME = rf"Dr\.?{AFTER_TITLE}{ONE_OR_MORE}"             # "Dr" + one or more words
NOT_CAPS = (r"(?!(?:GROSS|DESCRIPTION|FINAL|DIAGNOSIS|MICROSCOPIC|CLINICAL|HISTORY|SPECIMEN|COMMENT|"
            r"REPORT|FROZEN|SECTION|PATHOLOGY|NOTE|SEE|AND|THE|OF)\b)")
CAPS_NAME = rf"{NOT_CAPS}[A-Z][A-Z'\-]+(?:[ \t]+{NOT_CAPS}[A-Z][A-Z'\-]+){{0,3}}"   # "TAN AH KOW"
LSEP = r"[ \t]*(?:\([^)\n]{0,20}\))?[ \t:#.\-]*"           # label -> value: "DOB (dd/mm/yyyy): ", "MRN-"
D = r"[\dOl]"                                               # a digit, allowing OCR O for 0 and l for 1
TITLES = ("Mr", "Mrs", "Ms", "Mdm", "Madam", "Miss", "Sister")
MONTHS = r"(?i:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*)\.?"   # "Apr", "APR", "April"
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
# Title-case only: "pelvic floor" is anatomy, "FL 33101" is Florida, "Level 2 nodes" are neck levels.
FLOOR = r"(?:Floor|Fl\.?|Storey)"
# Newline-tolerant separator with disjoint branches. SEP's padded `\n` branch backtracked at digit starts
# and pushed the dev median over 100 ms here, so floor patterns use this instead.
GAP = r"(?:[ \t]+|\n)"
ORDINAL_WORD = r"(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth)"
MRN_OCR = r"[A-Z]{0,3}[\dO]{6,9}[A-Z]?"
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
PHONE_VALUE = (r"(?=(?:[^\dOlI\n]{0,2}[\dOlI]){7})\(?\+?\d[\dOlI \t().\-]{5,16}(?:\d|[OlI](?![A-Za-z]))"
               r"(?:[ \t]*(?:loc|ext)\.?[ \t]*\d+)?")   # trailing O/l/I only when a digit was misread, not a word ("loc")
# A bank by its name: "DBS Bank", "OCBC Bank Ltd", "Bank of America". Lower case "tumour bank" is a lab term, and
# so are these words before Bank: "Tissue Bank", "Tumor Bank", "Blood Bank".
NOT_BANK = r"(?!(?:Tissue|Tumou?r|Blood|Bio|Cord|Cell|Serum|DNA|RNA|Gene|Sample|Specimen|Data|Eye|Bone|Milk|Sperm|Food|Organ|Brain)\b)"
BANK_NAME = (rf"(?:{NOT_BANK}[A-Z][A-Za-z&'\-]+[ \t]+){{1,3}}Bank(?:[ \t]+(?:Berhad|Bhd|Ltd|Limited|plc|PLC|N\.A\.))?(?![A-Za-z])"
             r"|Bank[ \t]+of[ \t]+[A-Z][A-Za-z&'\-]+(?:[ \t]+[A-Z][A-Za-z&'\-]+){0,2}")
# A place by what it is: "Metropolitan Health", "... Cancer Ctr", "Foundation Medicine, Inc."
INST_WORD = r"(?:Ctr|Cent(?:er|re)|Hospital|Clinic|Institute|Health|Medical|Medicine|Lab(?:oratory|oratories)?|Infirmary)"
US_STATE = (r"(?:A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI|S[CD]|T[NX]|"
            r"UT|V[AT]|W[AIVY])")


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


class CardRecognizer(PatternRecognizer):
    """A payment card number with or without a label: 13-19 digits, in groups or not. 16 digits in
    4-4-4-4 groups are masked even with a bad checksum; other lengths must pass the Luhn check."""
    def __init__(self):
        super().__init__(supported_entity="ACCOUNT", name="card", global_regex_flags=FLAGS,
                         patterns=[Pattern("card", r"(?<![\d\-.])(?:\d{4}[ \-]+){3}\d{1,7}(?![\d\-])|(?<![\d\-.])\d{13,19}(?![\d\-.])", 0.6)])

    def validate_result(self, pattern_text: str) -> bool:
        if re.fullmatch(r"\d{4}(?:[ \-]+\d{4}){3}", pattern_text.strip()):
            return True                    # 16 digits in groups: mask even if mistyped
        return luhn_valid(pattern_text)


class IbanRecognizer(PatternRecognizer):
    """An IBAN with or without a label, kept only if its check digits are right."""
    def __init__(self):
        super().__init__(supported_entity="ACCOUNT", name="iban", global_regex_flags=FLAGS,
                         patterns=[Pattern("iban", r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b", 0.6)])

    def validate_result(self, pattern_text: str) -> bool:
        return iban_valid(pattern_text)


class ParticleNameRecognizer(EntityRecognizer):
    """A whole name around a particle, in any case: "Nur Aisyah binte Rahman", "mohd rizal bin harun",
    "Kavitha d/o Rajendran". Words before the particle that are ordinary report words ("the sample from")
    are trimmed. Lower-case "bin" is also an English word ("sharps bin"), so a name needs a word next to it
    that is not a report word; "van"/"von" stay Title-case only."""
    # one space between the words of a name: a wider gap is a column ("... binte Rashid    NRIC: ...")
    RX = re.compile(r"(?<![A-Za-z'])((?:[A-Za-z][A-Za-z'\-]* ){0,3})"
                    r"(binte|binti|bte|bin|s/o|d/o|a/l|a/p|van der|van|von) "
                    r"([A-Za-z][A-Za-z'\-]+(?: [A-Za-z][A-Za-z'\-]+)?)(?![A-Za-z'])", re.IGNORECASE)

    def __init__(self):
        super().__init__(supported_entities=["NAME"], name="particle_name")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        from deid.hold import name_list, vocabulary
        vocab, names = vocabulary(), name_list()
        ordinary = lambda w: w.lower() in vocab and w.lower() not in names
        out = []
        for m in self.RX.finditer(text):
            before, particle, after = m.group(1).split(), m.group(2), m.group(3).split()
            if particle.lower() in ("van", "von", "van der") and not after[0][0].isupper():
                continue                                              # "van der Meer", not "a van to the clinic"
            if any(w[0].isupper() for w in before + after):           # a Title-case name: only its capitalised words
                k = len(before)
                while k and before[k - 1][0].isupper():
                    k -= 1
                before = before[k:]                                   # "husband Ahmad Faizal", "reviewed by van der"
            while before and ordinary(before[0]):
                before.pop(0)                                         # "the sample from" is not part of the name
            if particle.lower() == "bin" and (not before or ordinary(before[-1])) and ordinary(after[0]):
                continue                                              # "sharps bin by the door"
            if ordinary(after[0]):
                continue
            # a second word after the particle only if it looks like a name too: capitalised in a Title-case
            # name ("binte Abdul Rahman"), or a listed name in a lower-case one; never "asked", "arrived"
            titled = after[0][0].isupper()
            if len(after) == 2 and (ordinary(after[1]) or (not after[1][0].isupper() if titled else after[1].lower() not in names)
                                    or after[1].isupper()):
                after = after[:1]
            start = text.rfind(before[0], m.start(), m.start(2)) if before else m.start(2)
            end = text.find(after[-1], m.start(3)) + len(after[-1])
            out.append(RecognizerResult("NAME", start, end, 0.65, recognition_metadata={"rule": "particle_name"}))
        return out


class ReversedNameRecognizer(EntityRecognizer):
    """Surname-first names: 'Al-Mansoor, Tariq', 'TAN, K. W.'. Kept only if the surname half is listed."""

    RX = re.compile(
        rf"(?P<v>{NOT_CAPS}[A-Z][A-Za-z'\-]{{1,29}},[ \t]+(?:[A-Z]\.(?:[ \t]*[A-Z]\.){{0,2}}|[A-Z][A-Za-z'\-]{{2,}}))"
        r"(?![A-Za-z'\-])", FLAGS)

    def __init__(self):
        super().__init__(supported_entities=["NAME"], name="reversed_name")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        from deid.hold import SURNAMES, name_list, vocabulary
        names, vocab = name_list(), vocabulary()
        out = []
        for m in self.RX.finditer(text):
            first, given = (x.strip().lower() for x in m.group("v").split(",", 1))
            if given in vocab and given not in names or re.sub(r"[^a-z]", "", given) in ("md", "do", "phd", "mb", "bs"):
                continue                                    # "Hassan, Consultant", "Ramkissoon, M.D.": a title or degree
            parts = set(first.replace("-", " ").split()) | {first}
            if any((w in names and w not in vocab) or w in SURNAMES for w in parts):
                out.append(RecognizerResult("NAME", m.start("v"), m.end("v"), 0.7,
                                            recognition_metadata={"rule": "reversed_name"}))
        return out


class AddressBlockRecognizer(EntityRecognizer):
    """A mailing block, as on a statement or letter: a line that is only a name, then 1-5 short lines down to a
    postcode line ("SG 999999", "Singapore 520088", "50450 Kuala Lumpur"). The name is a NAME, each line an ADDRESS.
    A line may carry a second column after 3+ spaces (PDF layout): only the first column counts."""
    NAME_LINE = re.compile(r"(?:(?:Mr|Mrs|Ms|Mdm|Dr|MR|MRS|MS|MDM|DR)\.?[ \t]+)?"
                           r"[A-Z][A-Za-z'\-]+(?:[ \t][A-Z][A-Za-z'\-]+){1,3}")
    PART = re.compile(r"[A-Za-z0-9#][A-Za-z0-9 ,.'#/\-]{0,44}")
    POSTCODE = re.compile(r"(?:(?:[A-Z]{2}|[A-Za-z][A-Za-z .]+)[ \t]+)?\d{5,6}|\d{5,6}[ \t]+[A-Za-z][A-Za-z .]+")

    def __init__(self):
        super().__init__(supported_entities=["NAME", "ADDRESS"], name="address_block")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        from deid.hold import vocabulary

        lines, pos = [], 0
        for line in text.split("\n"):
            lead = len(line) - len(line.lstrip(" \t"))
            first = re.split(r"[ \t]{3,}", line.strip(" \t"))[0] if line.strip() else ""
            lines.append((pos + lead, first))
            pos += len(line) + 1
        out = []
        for i, (start, name) in enumerate(lines):
            if not self.NAME_LINE.fullmatch(name) or all(w.lower() in vocabulary() for w in re.findall(r"[A-Za-z']+", name)):
                continue
            block = []
            for s, part in lines[i + 1:i + 6]:
                if not part or not self.PART.fullmatch(part):
                    break
                block.append((s, part))
                if self.POSTCODE.fullmatch(part):
                    out.append(RecognizerResult("NAME", start, start + len(name), 0.7,
                                                recognition_metadata={"rule": "address_block"}))
                    out += [RecognizerResult("ADDRESS", s, s + len(p), 0.7, recognition_metadata={"rule": "address_block"})
                            for s, p in block]
                    break
        return out


class AddressPartsRecognizer(EntityRecognizer):
    """An address by its parts, as an address parser reads it: unit, street, building or estate, postcode, and
    city / state / country. Two or more different parts next to each other (a comma, spaces or one line break
    between) make one ADDRESS; one part alone never does, so "Level 2 nodes", "Hill's criteria" or "Park et al."
    stay. At least one part must be a unit, street or building, unless it is a postcode with a place."""
    CAP = r"[A-Z][A-Za-z'\-]*"
    PARTS = {
        "unit": r"#\d{1,3}-\d{1,5}[A-Z]?|(?:Unit|Apt|Apartment|Flat|Suite|Ste)\.?[ \t]?#?\d{1,4}[A-Z]?(?:-\d{1,5})?"
                r"|Basement(?:[ \t][A-Z][a-z]+)?[ \t]\d{1,2}\b"
                r"|(?:Level|Lvl)[ \t]\d{1,3}(?=[ \t]*,)",
        "street": rf"(?:(?:Blk|Block|No\.?)[ \t]?)?(?:\d{{1,5}}[A-Z]?,?[ \t]+)?(?:{CAP}[ \t]+){{1,3}}"
                  r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Close|Lane|Ln|Way|Crescent|Cres|Boulevard|Blvd|Terrace|Tce|"
                  r"Place|Walk|Link|Rise|Grove|Gateway|Highway|Hwy|Square|Loop|Vale)\b\.?(?:[ \t]\d{1,2}\b)?"
                  rf"|(?:\d{{1,5}}[A-Z]?,?[ \t]+)?(?:Jalan|Jln|Lorong|Lor|Persiaran|Lebuh|Lebuhraya|Taman)\.?(?:[ \t]+{CAP}){{1,3}}(?:[ \t]\d{{1,2}}\b)?",
        "building": rf"(?:The[ \t]+)?(?:{CAP}[ \t]+){{1,3}}(?:Residences|Residence|Towers|Tower|Court|Gardens|Condominium|"
                    r"Plaza|Mansions|Apartments|Estate|Heights|Suites|Building|Bldg)\b",
        "postcode": r"(?<![\d\-.])\d{5,6}(?![\d\-]|\.\d)",
        "place": r"Singapore|Malaysia|Kuala[ \t]Lumpur|Petaling[ \t]Jaya|Shah[ \t]Alam|Subang[ \t]Jaya|Selangor|Johor[ \t]Bahru|"
                 r"Johor|Penang|Pulau[ \t]Pinang|Perak|Ipoh|Melaka|Malacca|Kedah|Pahang|Negeri[ \t]Sembilan|Putrajaya|Sabah|"
                 r"Sarawak|Kuching|Indonesia|Jakarta|Batam|Thailand|Bangkok|Hong[ \t]Kong|United[ \t]Kingdom|"
                 r"United[ \t]States|USA|Australia",
    }
    PART = re.compile("|".join(f"(?P<{k}>{v})" for k, v in PARTS.items()))
    JOIN = re.compile(r"[ \t]*[,;]?[ \t]*(?:\n[ \t]*)?")

    def __init__(self):
        super().__init__(supported_entities=["ADDRESS"], name="address_parts")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        out, chain = [], []

        def close():
            kinds = {k for k, _, _ in chain}
            if len(kinds) >= 2 and (kinds & {"unit", "street", "building"} or kinds == {"postcode", "place"}):
                out.append(RecognizerResult("ADDRESS", chain[0][1], chain[-1][2], 0.75,
                                            recognition_metadata={"rule": "address_parts"}))

        for m in self.PART.finditer(text):
            if chain and not self.JOIN.fullmatch(text[chain[-1][2]:m.start()]):
                close()
                chain = []
            chain.append((m.lastgroup, m.start(), m.end()))
        if chain:
            close()
        return out


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
        ("NAME", "name_label", rf"(?:^|\|)[ \t]*(?i:name|patient name)[ \t]*[:|][ \t]*"
                               rf"(?P<v>{WORD},[ \t]+{ONE_OR_MORE}|{NAME}|{CAPS_NAME})"),   # also "Surname, Given Names"
        ("NAME", "relation_before", rf"\b{RELATIONS}(?:-in-law)?,?[ \t]+(?P<v>{ONE_OR_MORE})"),
        ("NAME", "relation_after", rf"(?P<v>{NAME})[ \t]*\({RELATIONS}(?:-in-law)?\)"),
        ("NAME", "footer", rf"^Page \d+ of \d+ *\| *(?P<v>{NAME}) *\|"),
        ("DOCTOR", "dr", rf"\b(?P<v>{DR_NAME})"),
        ("DOCTOR", "prof", rf"\bProf(?:essor)?\.?{AFTER_TITLE}(?P<v>{ONE_OR_MORE})"),
        ("NAME", "title", rf"\b(?:{'|'.join(TITLES)})\.?{AFTER_TITLE}(?P<v>{ONE_OR_MORE})"),
        ("DOCTOR", "dr_caps", rf"\b(?P<v>(?i:dr|prof)\.?[ \t]+{CAPS_NAME})\b"),
        ("DOCTOR", "dr_apos", rf"\b(?:Dr|Prof(?:essor)?)\.?{AFTER_TITLE}"
                              r"(?P<v>[A-Za-z]+'[A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)?)"),
        ("NAME", "title_caps", rf"\b(?i:mr|mrs|ms|mdm|madam|miss)\.?[ \t]+(?P<v>{CAPS_NAME})\b"),
        ("ADDRESS", "address", r"^(?:Address|Addr\.?|ADDR\.?):[ \t]*(?P<v>[^\n]*[^\s])"),
        ("HOSPITAL", "hospital", rf"\b(?P<v>{HOSPITAL})\b"),
        # a letterhead in capitals: "HARBOURVIEW GENERAL HOSPITAL", "ST ANNE'S MEDICAL CENTRE"
        ("HOSPITAL", "hospital_caps", r"\b(?P<v>(?:ST\.? )?(?:[A-Z][A-Z'\-]+ ){1,4}(?:GENERAL HOSPITAL|HOSPITAL|"
                                      r"CANCER (?:INSTITUTE|CENT(?:ER|RE))|MEDICAL CENT(?:ER|RE)|INFIRMARY))\b"),
        # "Facility: Metropolitan Health" + "Cancer Ctr" on the next line: one name, wrapped
        ("HOSPITAL", "facility", r"^[ \t]*(?i:facility|institution|hospital|clinic|referring (?:facility|hospital))[ \t]*:"
                                 rf"[ \t]*(?P<v>[^\n:]*[^\s:](?:\n[ \t]*(?:[A-Z][A-Za-z&'\-]*[ \t]+){{0,4}}{INST_WORD}\.?(?=[ \t]*$))?)"),
        ("HOSPITAL", "company", r"\b(?P<v>(?:[A-Z][A-Za-z&'\-]+[ \t]+){0,3}[A-Z][A-Za-z&'\-]+,?[ \t]+"
                                r"(?:Inc|LLC|Ltd|Corp|Corporation|GmbH|Pte\.?[ \t]+Ltd)\b\.?)"),
        ("BANK", "bank", rf"\b(?P<v>{BANK_NAME})"),
        ("ID", "clia", r"\bCLIA(?:[ \t]*(?i:no\.?|number|#))?[ \t]*[:#.]?\s*(?P<v>\d{2}[D\d]\d{7})(?![\dA-Za-z])"),
        # DOB in any format: dates are kept by default, so an unlabelled DOB would stay in the text.
        ("DOB", "dob", rf"\b(?i:DOB|D\.O\.B\.?|Date of birth|Birth ?date|Born|b\.)(?:[ \t]*/[ \t]*(?i:age))?{LSEP}(?:on[ \t]+)?"
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
        ("ID", "lab_ref", r"\b(?i:block|cassette|slide|specimen|barcode|reference)\b"
                          r"(?:[ \t]+(?:marked|labelled|labeled|stamped|numbered))?"
                          r"[ \t]*:?[ \t]*(?P<v>[A-Z0-9][A-Z0-9/#-]{3,24})(?![A-Z0-9/#-])"),
        ("PHONE", "phone_cue", rf"\b{PHONE_CUE}[ \t:#.]*(?P<v>{PHONE_VALUE})(?!\d)"),
        ("PHONE", "pager", rf"\b(?P<v>(?i:b/p|pager)[ \t:#.]*{PHONE_VALUE})(?!\d)"),
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
    DIGIT_RULES = {"id_label", "account_any", "lab_ref"}

    def analyze(self, text: str, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        from deid.hold import vocabulary

        out = []
        for entity, rule, rx in self.rules:
            for m in rx.finditer(text):
                value = re.sub(r"^(?i:dr|prof)\.?\s+", "", m.group("v"))
                if rule in self.DIGIT_RULES and sum(ch.isdigit() for ch in value) < (5 if rule == "lab_ref" else 3):
                    continue                              # "NODES EXAMINED", "AGE/SX", cassette "A1-16": not an identifier
                if rule in self.CAPS_RULES and value.isupper() and all(
                        w.lower() in vocabulary() for w in re.findall(r"[A-Za-z']+", value)):
                    continue
                kind = "BANK" if rule == "company" and re.search(r"\bBank\b", value) else entity   # "OCBC Bank Ltd."
                out.append(RecognizerResult(kind, m.start("v"), m.end("v"), 0.85,
                                            recognition_metadata={"rule": f"labelled_{rule}"}))
        return out


RECOGNISERS = [
    NricRecognizer(),
    NhsRecognizer(),
    CardRecognizer(),
    IbanRecognizer(),
    ReversedNameRecognizer(),
    AddressBlockRecognizer(),
    AddressPartsRecognizer(),
    simple("postcode_ref", "ADDRESS", 0.6, r"(?<=\()\d{6}[A-Z](?=\))"),   # a statement's "(999999B)": postcode + letter
    simple("long_number", "ACCOUNT", 0.5, r"(?<![\d\-.])\d{12,19}(?![\d\-.])"),   # a statement's account number
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
           r"\b\d{3}-\d{3}-\d{4}\b",                                # US 555-231-8846
           r"(?<![\d.])(?:1\.)?\d{3}\.\d{3}\.\d{4}(?![\d.])"),       # US 1.888.988.3639
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
           rf"\b[A-Z][a-z]+(?:[ \t][A-Z][a-z]+){{0,2}},[ \t]+{US_STATE}[ \t]+\d{{5}}(?:-\d{{4}})?\b",   # Cambridge, MA 02141
           r"\bS\(?\d{6}\)?(?!\d)",                                  # SG postal code S520088
           r"\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b"),                   # UK postcode NE1 4XY
    simple("mrn", "MRN", 0.9, r"\bMRN\d{6,8}[A-Z]?\b"),
    ParticleNameRecognizer(),
    # S P A C E D  letters: "T A N  K O H  W A H" (single capitals, one space inside a word, two between words)
    simple("spaced_name", "NAME", 0.6, r"(?<![A-Za-z])[A-Z](?: [A-Z]){2,}(?:  [A-Z](?: [A-Z])+)+(?![A-Za-z])"),
    # a barcode line prints IDs letter-spaced: "* S 1 2 3 4 5 6 7 D *", "P A 3 0 - 1 2 3 4 5" (5+ digits)
    simple("spaced_id", "ID", 0.6, r"(?<![\w-])(?=(?:[A-Z\-] )*(?:\d[ \-]+){4}\d)[A-Z0-9](?: [A-Z0-9\-]){6,}(?![\w-])"),
    simple("phone_ext", "PHONE", 0.6, r"\b(?:ext|extn)\.?[ \t]?\d{3,5}\b|(?<![\w-])x\d{4,5}\b"),
    simple("station", "ADDRESS", 0.6, r"\b(?:[A-Z][a-z]+[ \t]){1,3}(?:MRT|LRT)(?:[ \t]Station)?\b"),
    simple("dotted_name", "NAME", 0.6, r"\b(?!(?:Mr|Mrs|Ms|Mdm|Dr|Prof|St)\.)[A-Za-z]{2,}\.[A-Z][a-z]+"),   # not "Mr.Tan"
    simple("surg_acc", "ACCESSION", 0.8, r"\bSURG-?\d{4}-?\d{4,}\b"),
    simple("floor", "ADDRESS", 0.8,
           rf"\b\d{{1,3}}(?:st|nd|rd|th)?{GAP}{FLOOR}\b",            # "1st Floor", "2 Fl", "3 Storey", "1st\nFloor"
           rf"\b{FLOOR}{GAP}\d{{1,3}}\b",                            # "Floor 1", "Fl 2"
           rf"\b(?:{ORDINAL_WORD}{GAP}{FLOOR}|Ground{GAP}Floor)\b"),     # "First Floor", "Ground Floor"
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


_UNIT = (r"(?:\d{1,3}(?:st|nd|rd|th)?[ \t]+floor|level[ \t]+\d{1,3}|suite[ \t]+\d+|unit[ \t]+\d+"
         r"|#\d{1,4}-\d{1,4}|blk[ \t]+\d+)")
_GAP = re.compile(rf"[,. \t]*(?:{_UNIT}[, \t][,. \t]*)?", re.IGNORECASE)


def merge_addresses(text, kept):
    """Join ADDRESS spans split only by punctuation or one unit part ("1st Floor", "Suite 200")."""
    out, first, cur = [], None, None
    for source, r in kept:
        if r.entity_type == "ADDRESS" and cur is not None and _GAP.fullmatch(text[cur.end:r.start]):
            cur = RecognizerResult("ADDRESS", cur.start, r.end, max(cur.score, r.score))
        else:
            if cur is not None:
                out.append((first, cur))
                first, cur = None, None
            if r.entity_type == "ADDRESS":
                first, cur = source, r
            else:
                out.append((source, r))
    if cur is not None:
        out.append((first, cur))
    return out


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

    Scans a folded copy (look-alike characters made plain) with invisible characters taken out, so "Ta​n"
    or "Sa­muel" read as one word; every span is mapped back to the original text.
    """
    with timed("presidio", report_id=report_id, chars=len(text)):
        keep = [i for i, ch in enumerate(text) if ch not in INVISIBLE]
        scan = fold("".join(text[i] for i in keep))
        found = _find(scan, learned, names_model, report_id)
        if len(keep) < len(text):                                     # back to offsets in the original
            keep.append(len(text))
            found = [(t, keep[s], keep[e - 1] + 1, source, score) for t, s, e, source, score in found]
    return [
        PhiSpan(id=f"p{i}", type=t, start=s, end=e, text=text[s:e], source=source, score=score)
        for i, (t, s, e, source, score) in enumerate(sorted(found, key=lambda x: x[1]), 1)
    ]


INVISIBLE = set("­​‌‍⁠﻿")      # soft hyphen and zero-width characters


def _find(scan: str, learned, names_model: bool, report_id) -> list[tuple]:
    """(type, start, end, source, score) for every identifier in the scan text."""
    results = []
    for rec in RECOGNISERS + learned_recognisers(learned):
        for r in rec.analyze(scan, rec.supported_entities, nlp_artifacts=None):
            rule = (r.recognition_metadata or {}).get("rule") or rec.name
            results.append((f"pattern:{rule}", r))
    kept = merge_addresses(scan, resolve_overlaps(results))
    found = [(r.entity_type, r.start, r.end, source, round(r.score, 2)) for source, r in kept]
    with timed("ner", report_id=report_id, chars=len(scan)):    # the model only fills gaps
        found += [] if not names_model else [("NAME", a, b, "ner:spacy", 0.6) for a, b in find_names(scan)
                  if all(b <= s or a >= e for _, s, e, _, _ in found)]
    found += propagate_names(scan, found)
    found += propagate_postcodes(scan, found)
    return found


def propagate_postcodes(text: str, found: list[tuple]) -> list[tuple]:
    """Same report: once an address holds a 5-6 digit postcode, its other mentions are masked too, bare or with
    a letter after it, as a statement prints it in a reference ("(999999B)", "999999 0000111122223333444")."""
    taken = [(s, e) for _, s, e, _, _ in found]
    codes = {c for t, s, e, _, _ in found if t == "ADDRESS" for c in re.findall(r"(?<!\d)\d{5,6}(?!\d)", text[s:e])}
    extra = []
    for code in codes:
        for m in re.finditer(rf"(?<![\dA-Za-z]){code}[A-Z]?(?![\dA-Za-z])", text):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                extra.append(("ADDRESS", m.start(), m.end(), "pattern:repeat_postcode", 0.8))
                taken.append((m.start(), m.end()))
    return extra


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
