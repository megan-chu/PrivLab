"""
ner.py - person names from a small local NER model (spaCy, on this device), for names without a cue
("Escalated to Marcus Wong"). Only spaCy's NER component runs (~70 ms for a typical report).

Filtered so pathology wording is not flagged: a span in ALL CAPS, with a digit, or made only of report
vocabulary ("Consultant Pathologist", "Lymph Nodes") is dropped. Setting PRIVLAB_NER_MODEL in .env
(default en_core_web_sm; "off" disables). A missing model is logged once and skipped, never a crash.

Usage: from deid.ner import find_names; find_names(text)   # [(start, end), ...]
"""
import logging
import re
from functools import lru_cache

from common.config import load_config

log = logging.getLogger("privlab.ner")
PARTICLES = {"bin", "binte", "binti", "bte", "s/o", "d/o", "a/l", "a/p", "van", "von", "de", "da"}
PARTICLE_RX = "|".join(re.escape(p) for p in sorted(PARTICLES, key=len, reverse=True))
KEEP_ONLY = ["ner"]
WINDOW = 80                                                # characters of context on each side
WORD = re.compile(r"\b[A-Z][a-z][A-Za-z'’\-]*")           # a capitalised word


@lru_cache(maxsize=1)
def _model():
    name = load_config().ner_model
    if not name or name.lower() == "off":
        return None
    try:
        import spacy
        nlp = spacy.load(name)
        return spacy.load(name, exclude=[p for p in nlp.pipe_names if p not in KEEP_ONLY])
    except Exception as e:                      # OSError if the model is not installed
        log.warning("NER model %s not available (%s); names come from patterns only", name, type(e).__name__)
        return None


def _looks_like_name(span: str, vocab) -> bool:
    from deid.hold import _plain

    words = re.findall(r"[A-Za-z][A-Za-z'’]*", span)                  # "Ziehl-Neelsen": both parts checked
    if not words or span.isupper() or re.search(r"\d", span):
        return False
    return any(len(w) > 1 and _plain(w) not in vocab for w in words if w[0].isupper())


def find_names(text: str) -> list[tuple[int, int]]:
    nlp = _model()
    if nlp is None:
        return []
    from deid.hold import LABEL_AFTER, _plain, vocabulary

    vocab, out = vocabulary(), []
    # Only text around a capitalised word that is not report vocabulary can hold a name: the model reads
    # those windows (on one line, merged), so even a long pathology report stays under 100 ms.
    windows = []
    for m in WORD.finditer(text):
        if _plain(m.group()) in vocab:
            continue
        line_start, line_end = text.rfind("\n", 0, m.start()) + 1, text.find("\n", m.end()) % (len(text) + 1)
        a, b = max(line_start, m.start() - WINDOW), min(line_end, m.end() + WINDOW)
        if windows and a <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(b, windows[-1][1]))
        else:
            windows.append((a, b))
    ents = [(a + e.start_char, a + e.end_char) for (a, b), doc in
            zip(windows, nlp.pipe(text[a:b] for a, b in windows)) for e in doc.ents if e.label_ == "PERSON"]
    for start, end in ents:
        if caps := re.search(r"[ \t]+[A-Z]{2,}\b", text[start:end]):
            end = start + caps.start()          # "Oliver Turner FROZEN SECTION" -> "Oliver Turner"
        while (m := re.search(r"[ \t]+(\S+)$", text[start:end])) and m.group(1).lower() in PARTICLES:
            end = start + m.start()             # "Nur Aisyah bte" -> "Nur Aisyah"
        end = start + len(text[start:end].rstrip(" \t-.,;:"))
        if end < len(text) and text[end].isalpha() or start and text[start - 1].isalpha():
            continue                            # cut mid-word ("Gross Descriptio|n"): not a name
        if LABEL_AFTER.match(text, end):
            continue                            # a form label ("Claim Ref:"), as in the hold rule
        if _looks_like_name(text[start:end], vocab):
            if out and re.fullmatch(rf"[ \t]+(?:{PARTICLE_RX})[ \t]+", text[out[-1][1]:start], re.IGNORECASE):
                out[-1] = (out[-1][0], end)     # "Nur Aisyah" + "bte" + "Rahman": one name
            else:
                out.append((start, end))
    return out
