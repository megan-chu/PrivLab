"""
hold.py - the hold rule: after the user confirms Layer 1 and before any API call, pause
on words in the masked text that still look like names.

Flags a run of 2+ Title-case words with at least one unknown word, or a single unknown
Title-case word mid-sentence. Ignores ALL-CAPS words (TCGA headings), lone sentence-initial
words, tags and the profile's allow-list. "Unknown" = not in the vocabulary: words from the
de-identified TCGA source text (dev + extra only, never test), standard report wording and
common eponyms.
"""
import csv
import re
import unicodedata
from functools import lru_cache

from common.data import DATA, manifest

EPONYMS = {"gleason", "nottingham", "fuhrman", "bloom", "richardson", "elston", "ellis", "hodgkin",
           "paget", "isup", "who", "tnm", "ajcc", "cap", "her", "ki", "dako", "hercep"}
REPORT_WORDS = {"department", "anatomical", "pathology", "surgical", "report", "consultant", "pathologist",
                "electronically", "signed", "requesting", "surgeon", "accession", "collected", "reported",
                "page", "patient", "comment", "results", "discussed", "phone", "address", "contact",
                "copy", "requested", "sent", "next", "kin", "emergency", "history", "seen", "clinic", "review"}
MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january",
          "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"}
TITLES = {"mr", "mrs", "ms", "mdm", "madam", "miss", "sister", "dr", "prof", "professor"}
PARTICLES = {"bin", "binte", "binti", "bte", "s/o", "d/o", "a/l", "a/p"}
SURNAMES = {"tan", "hope", "rose", "mark"}
# Common surnames that are also ordinary English words: a Title-case mention flags for review
# even though the word is in the vocabulary. Sentence-start and allow-list exemptions still apply.
# Kept only words that are rarely Title-case mid-sentence in clinical text. Dropped young, will,
# long, hall, bell, stone: they held "Young patient", "Will repeat", "Long axis", "In Hall B";
# Muse Spark still checks those names as the second layer.
# No "low" (tumour grade, Title-case after labels: "Nuclear grade: Low") and no colour words
# (white/black/brown/green: "A: White fibrous tissue"); both false-alarmed on dev/extra.
LETTER = r"[^\W\d_]"                                   # any letter: José, Muñoz
TOKEN = re.compile(rf"(?:s/o|d/o|a/l|a/p)(?![A-Za-z])|{LETTER}(?:{LETTER}|['’\-])*")
TAG = re.compile(r"\[[A-Z]+-\d+\]")
INITIALS_BEFORE = re.compile(r"(?:\b[A-Z]\.[ \t]*)+$")   # "L. " before "Kumar"
INITIALS_AFTER = re.compile(r"^(?:[ \t]*[A-Z]\.)+")       # " B. H." after "Lim"
WORDPART = rf"{LETTER}(?:{LETTER}|['’\-])*"
PARTICLE_NAME = re.compile(rf"(?<![A-Za-z'])({WORDPART})[ \t]+"
                           rf"(binte|binti|bin|bte|s/o|d/o|a/l|a/p)[ \t]+"
                           rf"({WORDPART})(?![A-Za-z'])", re.IGNORECASE)
# "X binte Y" in any case ("siti binte ali"); "de/da/van/von" are too noisy in lower case.


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    allowed_files = {r["tcga_file"] for r in manifest() if r["split"] in ("dev", "extra")}
    words = set(EPONYMS) | REPORT_WORDS | MONTHS | TITLES
    with open(DATA / "raw" / "tcga_starter_200.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["patient_filename"] in allowed_files:
                words.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*", row["text"]))
    return frozenset(words)


def _plain(word: str) -> str:
    """Lower case without accents: "Surgiçal" (OCR) -> "surgical"."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))


def _unknown(word: str, vocab, allowed) -> bool:
    if word.lower() in allowed:
        return False
    if word.lower().replace("’", "'").removesuffix("'s") in SURNAMES:          # "Tan", "Tan's"
        return True
    if _plain(word).replace("’", "'") in vocab:    # "Gerota's" as a whole
        return False
    parts = [p for p in re.split(r"[-’']", word) if len(p) > 1]          # "O’Brien" -> "Brien"
    return any(_plain(p) not in vocab for p in parts)


def _title(word: str) -> bool:
    return len(word) >= 2 and word[0].isupper() and not word.isupper()


def _sentence_start(text: str, pos: int) -> bool:
    """Only a new line or . ! ? start a sentence; "Consultant: Kumar", "L. Kumar", "Dr. Kumar" do not."""
    before = text[:pos].rstrip(" \t")
    if not before or before.endswith("\n") or before[-1] in "!?":
        return True
    if before[-1] == ".":
        prev = re.search(r"(\S+)\.$", before)
        word = prev.group(1) if prev else ""
        return not ((len(word) == 1 and word.isalpha()) or word.lower() in TITLES)
    return False


def hold_flags(text: str, allowed=(), tags=None) -> list[tuple[int, int]]:
    """Spans (offsets in the given masked text) that look like names.

    tags: PrivLab's own TagSpans for this text. Bracketed tokens that look like tags but are not
    ours (e.g. "[ONGKOKWAH-1]" typed into a report) are flagged, not skipped.
    """
    from deid.normalise import fold

    text = fold(text)                     # Cyrillic look-alikes etc.; same length, so offsets hold
    vocab, allowed = vocabulary(), {a.lower() for a in allowed}
    if tags is None:
        tag_spans = [(m.start(), m.end()) for m in TAG.finditer(text)]
        fake = []
    else:
        tag_spans = [(t.start, t.end) for t in tags]
        fake = [(m.start(), m.end()) for m in TAG.finditer(text) if (m.start(), m.end()) not in tag_spans]
    tokens = [m for m in TOKEN.finditer(text)
              if not any(a <= m.start() < b for a, b in tag_spans + fake)]

    runs, current = [], []
    for m in tokens:
        word = m.group()
        joins = current and text[current[-1].end():m.start()] in (" ", "\t")
        if _title(word) or (current and joins and word in PARTICLES):
            if current and not joins:
                runs.append(current)
                current = []
            current.append(m)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    flags = []
    for run in runs:
        words = [m for m in run if m.group() not in PARTICLES]
        if not words or not any(_unknown(m.group(), vocab, allowed) for m in words):
            continue
        if len(words) == 1 and _sentence_start(text, words[0].start()):
            continue
        start, end = run[0].start(), run[-1].end()
        if m := INITIALS_BEFORE.search(text[:start]):     # include initials, so none are left visible
            start = m.start()
        if m := INITIALS_AFTER.match(text[end:]):
            end += m.end()
        flags.append((start, end))

    for m in PARTICLE_NAME.finditer(text):
        if any(a <= m.start() < b or a < m.end() <= b for a, b in tag_spans):
            continue                                        # already masked
        flankers = (m.group(1), m.group(3))
        if any(w.lower() in allowed for w in flankers):
            continue                                        # user confirmed this word is not a name
        if not any(_unknown(w, vocab, allowed) for w in flankers):
            continue                                        # "waste bin liner": ordinary words around "bin"
        start, end = m.start(), m.end()
        for a, b in [f for f in flags if not (f[1] <= start or f[0] >= end)]:
            start, end = min(start, a), max(end, b)         # union: no part of the name stays visible
            flags.remove((a, b))
        flags.append((start, end))
    return sorted(flags + fake)
