"""
pdf.py - text from a PDF that has a text layer, on this device. Scanned PDFs (pages that are
only images) are refused: PrivLab does no OCR, and report content never goes to a cloud API.

Usage: from common.pdf import pdf_text; out = pdf_text(data)   # {"text", "warnings"} or {"error"}
"""
import io
import logging
import re
import textwrap

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 10_000_000
MIN_CHARS = 20                    # less than this is not a report: a scan or an empty file
NOT_SUPPORTED = "Scanned PDFs, images and other formats are not supported."
VERTICAL = re.compile(r"(?:^[ \t]*\S[ \t]*\n){3,}[ \t]*\S[ \t]*$", re.MULTILINE)   # 4+ lines of one character each
# A table row: words, a wide gap, and an amount at the end ("01 OCT   GRAB SINGAPORE     12.50 CR")
TABLE_ROW = re.compile(r"^.*\S[ \t]{3,}.*\d[\d,]*\.\d{2}(?:[ \t]*(?:CR|DR|-))?[ \t]*$", re.MULTILINE)
TABLE_ROWS = 5

logging.getLogger("pypdf").setLevel(logging.ERROR)     # malformed-PDF chatter; never to stdout


# Typeset ligatures as their letters ("reﬂex" -> "reflex"): otherwise the word is unknown to the word list, and two
# such words in a row look like a lower-case name to the de-identification rules.
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
                           "ﬅ": "st", "ﬆ": "st"})


def _clean(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").translate(LIGATURES)
    text = VERTICAL.sub(lambda m: "".join(m.group().split()), text)   # margin text, one character per line
    text = textwrap.dedent("\n".join(line.rstrip() for line in text.split("\n")))   # no page-margin indent
    return re.sub(r"\n{3,}", "\n\n", text)


CELL = re.compile(r"\S+(?:[ \t]{1,11}\S+)*")               # words with gaps under 12 spaces: one column's cell


def _tidy(line: str) -> str:
    """Layout mode spreads one cell's words apart ("JANE    SAMPLE   DOE"): one space between them, and each cell
    still starts at its own column, so the columns stay aligned."""
    out = ""
    for m in CELL.finditer(line):
        out = out.ljust(m.start()) if len(out) < m.start() else out + "  "
        out += re.sub(r"[ \t]+", " ", m.group())
    return out


def _page_text(page) -> str:
    """Plain reading for prose (single spaces, lines that wrap); layout mode only on a page with a table
    (5+ rows ending in an amount), where it keeps the columns apart. Layout mode copies a justified
    paragraph's gaps, so prose read that way is ragged."""
    plain = _clean(page.extract_text())
    try:
        layout = _clean("\n".join(_tidy(line) for line in page.extract_text(extraction_mode="layout").split("\n")))
    except Exception:                                     # layout mode can fail on unusual fonts
        return plain
    return layout if len(TABLE_ROW.findall(layout)) >= TABLE_ROWS else plain


def pdf_text(data: bytes) -> dict:
    if not data.startswith(b"%PDF-"):
        return {"error": f"This file is not a PDF. {NOT_SUPPORTED}"}
    if len(data) > MAX_PDF_BYTES:
        return {"error": f"The PDF is too large ({len(data) // 1_000_000} MB; limit {MAX_PDF_BYTES // 1_000_000} MB)."}
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return {"error": "The PDF is password-protected. Remove the password and add it again."}
        pages = [_page_text(p) for p in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as e:
        return {"error": f"The PDF could not be read ({type(e).__name__}); it may be damaged. Only PDFs with selectable text are supported."}
    if sum(len(p.strip()) for p in pages) < MIN_CHARS:
        return {"error": f"This PDF has no selectable text: it looks scanned. {NOT_SUPPORTED}"}
    blank = [str(i) for i, p in enumerate(pages, 1) if not p.strip()]
    warnings = [f"Page{'s' if len(blank) > 1 else ''} {', '.join(blank)} had no selectable text "
                f"(scanned?) and {'were' if len(blank) > 1 else 'was'} skipped. Check the original."] if blank else []
    return {"text": "\n\n".join(p.strip("\n") for p in pages if p.strip()), "warnings": warnings}
