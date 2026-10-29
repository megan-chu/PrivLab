"""
fix_family_names.py - one-off (29 Oct, approved by the owner): make the planted names natural.

The first generator picked every name independently, so a "daughter" could have a male name, a
Malay woman could be "bin" (son of), siblings had different fathers and prostate patients had
female names. This rewrites only the planted NAME and EMAIL spans (and the relation word in the
comment where it contradicts the patient's sex), recomputes every answer-key offset, and
regenerates the demo PDFs. TCGA text and all other identifiers are untouched. Seeded per report.

Rules:
  patient sex   PRAD male; "female"/"male" in the text; BRCA female; otherwise random (seeded)
  Chinese       Surname Given. Children take the father's surname; spouses keep their own surname;
                siblings share the patient's surname.
  Malay         Given bin/binte FatherGiven. Children: bin/binte the father's given name; siblings
                share the patient's father; spouses have their own father.
  Indian/UK/US  Given Surname. Children and siblings share the patient's surname; the spouse
                shares it too (married name).
  relatives     daughter/sister/wife female; son/brother/husband male; husband/wife follows the
                patient's sex.
  emails        rebuilt from the patient's new name; digits and domain kept.

Usage:
  python -m scripts.fix_family_names            # rewrite data/reports, data/truth/phi, data/pdf
  python -m scripts.fix_family_names --check    # dry run: print what would change (dev + extra only)
"""
import argparse
import json
import random
import re
import sys

from common.data import DATA, PHI_TRUTH, REPORTS, load_phi_truth, load_report, manifest

# --------------------------------------------------------------------------
# Name pools (invented combinations; common names per community)
# --------------------------------------------------------------------------
CHINESE_SURNAMES = ["Tan", "Lim", "Lee", "Ng", "Ong", "Wong", "Goh", "Chua", "Chan", "Koh", "Teo", "Ang",
                    "Yeo", "Tay", "Ho", "Low", "Toh", "Sim", "Chong", "Chia"]
CHINESE_GIVEN = {"m": ["Wei Ming", "Jun Jie", "Zhi Hao", "Kok Wah", "Boon Huat", "Jian Hui", "Wei Jie",
                       "Chee Keong", "Kah Wai", "Yong Sheng", "Teck Seng", "Hock Chye"],
                 "f": ["Wei Ling", "Mei Xuan", "Hui Min", "Li Ting", "Siew Lan", "Pei Shan", "Xin Yi",
                       "Shu Fen", "Bee Leng", "Mei Ling", "Jia Hui", "Yan Ling"]}
MALAY_GIVEN = {"m": ["Muhammad Hafiz", "Ahmad Faizal", "Mohd Rizal", "Muhammad Irfan", "Azman", "Firdaus",
                     "Khairul Anwar", "Syafiq", "Zulkifli", "Hakim"],
               "f": ["Nurul Huda", "Nur Aisyah", "Siti Aminah", "Farah Nabila", "Nur Syafiqah", "Aisyah",
                     "Norhayati", "Siti Rohani", "Nurul Ain", "Zarina"]}
MALAY_FATHERS = ["Osman", "Abdullah", "Hassan", "Rahman", "Ismail", "Yusof", "Ibrahim", "Salleh", "Hamid",
                 "Jamal", "Kassim", "Rashid"]
INDIAN_GIVEN = {"m": ["Rajesh", "Suresh", "Vikram", "Arjun", "Karthik", "Ramesh", "Anand", "Prakash",
                      "Sanjay", "Ravi"],
                "f": ["Kavitha", "Ananya", "Lakshmi", "Priya", "Deepa", "Meena", "Divya", "Revathi",
                      "Shanti", "Anjali"]}
INDIAN_SURNAMES = ["Krishnan", "Nair", "Menon", "Pillai", "Subramaniam", "Raman", "Rao", "Iyer", "Kumar",
                   "Reddy"]
WESTERN_GIVEN = {"m": ["James", "Thomas", "William", "Oliver", "Daniel", "Michael", "David", "Robert",
                       "George", "Edward", "Richard", "Matthew"],
                 "f": ["Sophie", "Eleanor", "Emily", "Linda", "Patricia", "Karen", "Sarah", "Charlotte",
                       "Margaret", "Susan", "Helen", "Jennifer"]}
WESTERN_SURNAMES = {"uk": ["Holloway", "Pemberton", "Whitaker", "Clarke", "Walker", "Turner", "Hughes",
                           "Bennett", "Harris", "Fletcher", "Wright", "Thompson"],
                    "us": ["Garcia", "Johnson", "Miller", "Davis", "Martinez", "Anderson", "Taylor", "Moore",
                           "Jackson", "Lewis", "Thompson", "Clark"]}
RELATION_SEX = {"son": "m", "brother": "m", "husband": "m", "daughter": "f", "sister": "f", "wife": "f"}


class Person:
    def __init__(self, given: str, family: str, sex: str, locale: str):
        self.given, self.family, self.sex, self.locale = given, family, sex, locale

    @property
    def name(self) -> str:
        if self.locale == "sg_chinese":
            return f"{self.family} {self.given}"
        if self.locale == "sg_malay":
            return f"{self.given} {'bin' if self.sex == 'm' else 'binte'} {self.family}"   # family = father
        return f"{self.given} {self.family}"

    def email_local(self) -> str:
        if self.locale == "sg_chinese":
            return f"{self.family}.{self.given.split()[-1]}".lower()
        return f"{self.given.split()[0]}.{self.family}".lower()


def _pick(rng, pool, avoid=()):
    return rng.choice([x for x in pool if x not in avoid] or pool)


def new_person(rng, locale: str, sex: str, family: str | None = None, avoid_family=()) -> Person:
    if locale == "sg_chinese":
        return Person(_pick(rng, CHINESE_GIVEN[sex]), family or _pick(rng, CHINESE_SURNAMES, avoid_family), sex, locale)
    if locale == "sg_malay":
        return Person(_pick(rng, MALAY_GIVEN[sex]), family or _pick(rng, MALAY_FATHERS, avoid_family), sex, locale)
    if locale == "in":
        return Person(_pick(rng, INDIAN_GIVEN[sex]), family or _pick(rng, INDIAN_SURNAMES, avoid_family), sex, locale)
    pool = WESTERN_SURNAMES["uk" if locale == "uk" else "us"]
    return Person(_pick(rng, WESTERN_GIVEN[sex]), family or _pick(rng, pool, avoid_family), sex, locale)


def relative_of(rng, patient: Person, relation: str) -> Person:
    loc, sex = patient.locale, RELATION_SEX[relation]
    for _ in range(20):                                    # never the patient's own name
        if relation in ("son", "daughter"):
            if loc == "sg_malay":                          # bin/binte the father's given name
                father = patient.given if patient.sex == "m" else _pick(rng, MALAY_GIVEN["m"]).split()[-1]
                rel = new_person(rng, loc, sex, family=father)
            elif loc == "sg_chinese" and patient.sex == "f":   # children take the husband's surname
                rel = new_person(rng, loc, sex, avoid_family=(patient.family,))
            else:
                rel = new_person(rng, loc, sex, family=patient.family)
        elif relation in ("brother", "sister"):
            rel = new_person(rng, loc, sex, family=patient.family)
        elif loc in ("sg_chinese", "sg_malay"):            # spouse keeps their own surname / father
            rel = new_person(rng, loc, sex, avoid_family=(patient.family,))
        else:                                              # married name
            rel = new_person(rng, loc, sex, family=patient.family)
        if rel.given != patient.given:
            return rel
    return rel


def patient_sex(rng, cancer: str, text: str) -> str:
    if cancer == "PRAD":
        return "m"
    if re.search(r"\bfemale\b", text, re.I):
        return "f"
    if re.search(r"\bmale\b", text, re.I):
        return "m"
    if cancer == "BRCA":
        return "f"
    return rng.choice("mf")


# --------------------------------------------------------------------------
# Rewrite one report: edits -> new text and shifted answer key
# --------------------------------------------------------------------------
def rewrite(report_id: str):
    text, truth = load_report(report_id), load_phi_truth(report_id)
    rng = random.Random(f"42-family-{report_id}")
    locale, phi = truth["locale"], truth["phi"]
    doctors = {p["text"] for p in phi if p["type"] == "DOCTOR"}
    old_patient = re.search(r"Patient:[ \t]+(.+?)(?:[ \t]{2,}|\n)", text).group(1)

    sex = patient_sex(rng, truth["cancer_type"], text)
    for _ in range(50):                                    # not the same as any doctor in the report
        patient = new_person(rng, locale, sex)
        if not any(patient.name in d for d in doctors):
            break

    edits = []                                             # (start, end, new_text, phi index or None)
    for i, p in enumerate(phi):
        if p["type"] == "NAME" and p["text"] == old_patient:
            edits.append((p["start"], p["end"], patient.name, i))
        elif p["type"] == "NAME":
            m = re.search(r"Patient's (\w+) $", text[max(0, p["start"] - 30):p["start"]])
            relation = m.group(1)
            if relation in ("husband", "wife"):
                fixed = "husband" if patient.sex == "f" else "wife"
                if fixed != relation:
                    s = p["start"] - len(relation) - 1
                    edits.append((s, s + len(relation), fixed, None))
                    relation = fixed
            edits.append((p["start"], p["end"], relative_of(rng, patient, relation).name, i))
        elif p["type"] == "EMAIL":
            local, domain = p["text"].split("@")
            digits = re.search(r"\d*$", local).group()
            edits.append((p["start"], p["end"], f"{patient.email_local()}{digits}@{domain}", i))

    edits.sort()
    out, pos, delta, shift = [], 0, 0, []
    for s, e, new, i in edits:
        out.append(text[pos:s] + new)
        shift.append((s, delta, i, len(new)))
        delta += len(new) - (e - s)
        pos = e
    out.append(text[pos:])
    new_text = "".join(out)

    edited = {i: (s + d, n) for s, d, i, n in shift if i is not None}
    new_phi = []
    for i, p in enumerate(phi):
        q = dict(p)
        if i in edited:
            q["start"], length = edited[i]
            q["end"] = q["start"] + length
        else:
            before = [d_after for s, d_after in _deltas(edits) if s <= p["start"]]
            d = before[-1] if before else 0
            q["start"], q["end"] = p["start"] + d, p["end"] + d
        q["text"] = new_text[q["start"]:q["end"]]
        new_phi.append(q)
    for p, q in zip(phi, new_phi):                         # untouched identifiers keep their text
        if p["type"] not in ("NAME", "EMAIL"):
            assert p["text"] == q["text"], (report_id, p["type"])
    return new_text, {**truth, "phi": new_phi}, old_patient, patient


def _deltas(edits):
    """(edit start, cumulative delta after that edit) in order."""
    total, out = 0, []
    for s, e, new, _ in edits:
        total += len(new) - (e - s)
        out.append((s, total))
    return out


# --------------------------------------------------------------------------
# Demo PDFs: plain one-column text, standard library only
# --------------------------------------------------------------------------
def write_pdf(path, title: str, text: str) -> None:
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    lines = []
    for raw in text.split("\n"):
        while len(raw) > 95:
            cut = raw.rfind(" ", 0, 95)
            cut = cut if cut > 40 else 95
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    per_page = 64
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[]]
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]
    kids = []
    for n, page in enumerate(pages):
        body = [f"BT /F2 11 Tf 40 805 Td ({esc(title)}) Tj ET"] if n == 0 else []
        body += ["BT /F1 8.2 Tf 40 785 Td 11.6 TL"] + [f"({esc(l)}) '" for l in page] + ["ET"]
        stream = "\n".join(body).encode("cp1252", "replace")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {len(objects)} 0 R "
                       f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    data, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{i} 0 obj\n".encode() + (obj if isinstance(obj, bytes) else obj.encode()) + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    data += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(data))


def main() -> int:
    ap = argparse.ArgumentParser(description="Make planted names consistent (sex, family, culture).")
    ap.add_argument("--check", action="store_true", help="dry run; prints dev + extra changes only")
    args = ap.parse_args()
    rows = manifest()
    split_of = {r["report_id"]: r["split"] for r in rows}
    changed = 0
    for r in rows:
        rid = r["report_id"]
        new_text, new_truth, old, patient = rewrite(rid)
        for p in new_truth["phi"]:
            assert new_text[p["start"]:p["end"]] == p["text"], rid
        if args.check:
            if split_of[rid] != "test":
                rel = re.search(r"Patient's \w+ [^\n]*?(?= requested| called| asked|;|\.)", new_text)
                print(f"{rid} {new_truth['locale']:10} {r['cancer_type']} {old!r:32} -> {patient.name!r}"
                      + (f"  | {rel.group()}" if rel else ""))
            continue
        with open(REPORTS / f"{rid}.txt", "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        with open(PHI_TRUTH / f"{rid}.json", "w", encoding="utf-8", newline="") as f:     # LF, like data/
            f.write(json.dumps(new_truth, indent=2, ensure_ascii=False))
        changed += 1
    if not args.check:
        for pdf in sorted((DATA / "pdf").glob("PP*.pdf")):
            rid = pdf.stem.split("_")[0]
            write_pdf(pdf, "SURGICAL PATHOLOGY REPORT", load_report(rid))
        print(f"rewrote {changed} reports and answer keys; regenerated the demo PDFs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
