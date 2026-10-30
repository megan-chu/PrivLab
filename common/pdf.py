"""
pdf.py - text from a PDF that has a text layer, on this device. Scanned PDFs (pages that are
only images) are refused: PrivLab does no OCR, and report content never goes to a cloud API.

Usage: from common.pdf import pdf_text; out = pdf_text(data)   # {"text", "warnings"} or {"error"}
"""
import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 10_000_000
MIN_CHARS = 20                    # less than this is not a report: a scan or an empty file
NOT_SUPPORTED = "Scanned PDFs, images and other formats are not supported."

logging.getLogger("pypdf").setLevel(logging.ERROR)     # malformed-PDF chatter; never to stdout


def pdf_text(data: bytes) -> dict:
    if not data.startswith(b"%PDF-"):
        return {"error": f"This file is not a PDF. {NOT_SUPPORTED}"}
    if len(data) > MAX_PDF_BYTES:
        return {"error": f"The PDF is too large ({len(data) // 1_000_000} MB; limit {MAX_PDF_BYTES // 1_000_000} MB)."}
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return {"error": "The PDF is password-protected. Remove the password and add it again."}
        pages = [(p.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n") for p in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as e:
        return {"error": f"The PDF could not be read ({type(e).__name__}); it may be damaged. Only PDFs with selectable text are supported."}
    if sum(len(p.strip()) for p in pages) < MIN_CHARS:
        return {"error": f"This PDF has no selectable text: it looks scanned. {NOT_SUPPORTED}"}
    blank = [str(i) for i, p in enumerate(pages, 1) if not p.strip()]
    warnings = [f"Page{'s' if len(blank) > 1 else ''} {', '.join(blank)} had no selectable text "
                f"(scanned?) and {'were' if len(blank) > 1 else 'was'} skipped. Check the original."] if blank else []
    return {"text": "\n\n".join(p.strip("\n") for p in pages if p.strip()), "warnings": warnings}
