"""
hold.py - the hold rule: after the user confirms Layer 1 and before any API call, pause
on words in the masked text that still look like names.

Flags a run of 2+ Title-case words with at least one unknown word, or a single unknown
Title-case word mid-sentence. Ignores ALL-CAPS words (TCGA headings), lone sentence-initial
words, tags and the profile's allow-list. "Unknown" = not in the vocabulary: words from the
de-identified TCGA source text (dev + extra only, never test), standard report wording and
common eponyms.

Name list (data/names/: public Census surnames that are not also common English words, and
Census first names): a suspect made only of listed names that are not in the vocabulary ("Kumar consented",
"ming wei") is held here, locally, instead of being left to the model.
"""
import csv
import re
import unicodedata
from functools import lru_cache

from common.data import DATA, manifest

NAME_LISTS = (DATA / "names" / "surnames.txt", DATA / "names" / "first_names.txt")
USE_NAME_LIST = True     # eval/qualify_model.py turns this off to measure the model alone

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


@lru_cache(maxsize=1)
def name_list() -> frozenset[str]:
    lines = [w for p in NAME_LISTS for w in p.read_text(encoding="utf-8").splitlines()]
    return frozenset(w for w in lines if w and not w.startswith("#"))


def _listed_name(text: str, vocab) -> bool:
    """Every word (titles aside) is a listed name that is not also a report word."""
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text) if w.lower().rstrip(".") not in TITLES]
    return bool(words) and all(w in name_list() and _plain(w) not in vocab for w in words)


def _covered(length: int, spans) -> bytearray:
    """1 for each character inside one of the spans (fast lookups for long reports)."""
    out = bytearray(length)
    for a, b in spans:
        out[a:b] = b"" * (b - a)
    return out


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
    i = pos                                    # scan back, no copy of text[:pos] (long reports)
    while i and text[i - 1] in " \t":
        i -= 1
    if not i or text[i - 1] in "\n!?":
        return True
    if text[i - 1] == ".":
        k = i - 1
        while k and not text[k - 1].isspace():
            k -= 1
        word = text[k:i - 1]
        return not ((len(word) == 1 and word.isalpha()) or word.lower() in TITLES)
    return False


def hold_flags(text: str, allowed=(), tags=None) -> list[tuple[int, int]]:
    """Spans (offsets in the given masked text) that look like names: the Title-case rules plus
    suspects that are listed names."""
    flags = _title_flags(text, allowed, tags)
    if USE_NAME_LIST:
        from deid.normalise import fold
        folded, vocab = fold(text), vocabulary()
        flags += [(a, b) for a, b in _raw_suspects(text, allowed, tags, flags)
                  if _listed_name(folded[a:b], vocab)]
    return sorted(flags)


def _title_flags(text: str, allowed=(), tags=None) -> list[tuple[int, int]]:
    """Title-case runs, lone mid-sentence words, particle names and fake tags.

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
    busy = _covered(len(text), tag_spans + fake)
    tokens = [m for m in TOKEN.finditer(text) if not busy[m.start()]]

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


SUSPECT_RUN = re.compile(rf"(?<![A-Za-z'])({WORDPART})((?:[ \t]+{WORDPART}){{1,3}})(?![A-Za-z'])")


def suspects(text: str, allowed=(), tags=None) -> list[tuple[int, int]]:
    """Words the model must rule on (_raw_suspects minus listed names, which hold_flags holds)."""
    found = _raw_suspects(text, allowed, tags)
    if not USE_NAME_LIST:
        return found
    from deid.normalise import fold
    folded, vocab = fold(text), vocabulary()
    return [(a, b) for a, b in found if not _listed_name(folded[a:b], vocab)]


def _raw_suspects(text: str, allowed=(), tags=None, held=None) -> list[tuple[int, int]]:
    """Words the Title-case rules are unsure about and do not hold. Listed names among them are held
    (hold_flags); the rest go to the model as explicit questions ("is this a name?") and come back
    to the user unticked if the model says nothing.

      - a lone unknown Title-case word at the start of a sentence ("Kumar consented ...")
      - a surname-word at the start of a sentence ("Tan, a 54-year-old man ...")
      - 2-4 unknown words in a row in lower case or ALL CAPS ("lim boon huat", "ONG KOK WAH")

    Offsets are in the given (masked) text. Spans the Title-case rules flag (held, computed here
    if not given) are not repeated.
    """
    from deid.normalise import fold

    text = fold(text)
    vocab, allow = vocabulary(), {a.lower() for a in allowed}
    tag_spans = [(t.start, t.end) for t in tags] if tags is not None else \
        [(m.start(), m.end()) for m in TAG.finditer(text)]
    held = _title_flags(text, allowed, tags) if held is None else held

    busy = _covered(len(text), tag_spans + held)

    def free(start, end):
        return busy.find(1, start, end) < 0

    out = []
    for m in TOKEN.finditer(text):
        word = m.group()
        if not (_title(word) and _sentence_start(text, m.start()) and free(m.start(), m.end())):
            continue
        nxt = re.match(rf"[ \t]+({WORDPART})", text[m.end():])
        if nxt and _title(nxt.group(1)):
            continue                                            # part of a Title-case run: hold rule's job
        plain = word.lower().replace("’", "'").removesuffix("'s")
        if plain not in allow and (plain in SURNAMES or _unknown(word, vocab, allow)):
            out.append((m.start(), m.end()))
    # Runs of 2-4 unknown words in one case (all lower or ALL CAPS), after an optional title.
    tokens = [m for m in TOKEN.finditer(text) if free(m.start(), m.end())]
    run: list = []

    def close():
        names = [t for t in run if t.group().lower().rstrip(".") not in TITLES]
        if 2 <= len(names) <= 4:
            out.append((run[0].start(), run[-1].end()))

    for t in tokens:
        w = t.group()
        case_ok = w.islower() or (w.isupper() and len(w) > 1)
        joins = run and text[run[-1].end():t.start()] in (" ", "\t") and (
            w.islower() == run[-1].group().islower() or run[-1].group().lower() in TITLES)
        is_title_word = w.lower() in TITLES and not run
        unknown = case_ok and len(w) >= 2 and w.lower() not in allow and _unknown(w, vocab, allow)
        if (unknown or is_title_word) and (not run or joins):
            run.append(t)
        else:
            close()
            run = [t] if unknown or is_title_word else []
    close()
    return sorted(set(out))
