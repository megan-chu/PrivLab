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
TOKEN = re.compile(r"(?:s/o|d/o|a/l|a/p)(?![A-Za-z])|[A-Za-z][A-Za-z'\-]*")
TAG = re.compile(r"\[[A-Z]+-\d+\]")


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    allowed_files = {r["tcga_file"] for r in manifest() if r["split"] in ("dev", "extra")}
    words = set(EPONYMS) | REPORT_WORDS | MONTHS | TITLES
    with open(DATA / "raw" / "tcga_starter_200.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["patient_filename"] in allowed_files:
                words.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*", row["text"]))
    return frozenset(words)


def _unknown(word: str, vocab, allowed) -> bool:
    return word.lower() not in allowed and any(p and p.lower() not in vocab for p in word.split("-"))


def _title(word: str) -> bool:
    return len(word) >= 2 and word[0].isupper() and not word.isupper()


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \t")
    return not before or before[-1] in ".!?:;\n|" or before.endswith("\n")


def hold_flags(text: str, allowed=()) -> list[tuple[int, int]]:
    """Spans (offsets in the given masked text) that look like names."""
    vocab, allowed = vocabulary(), {a.lower() for a in allowed}
    tag_spans = [(m.start(), m.end()) for m in TAG.finditer(text)]
    tokens = [m for m in TOKEN.finditer(text) if not any(a <= m.start() < b for a, b in tag_spans)]

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
        flags.append((run[0].start(), run[-1].end()))
    return flags
