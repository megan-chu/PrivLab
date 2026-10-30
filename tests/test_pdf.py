"""Text PDFs are read on this device; scans, images, other formats and locked PDFs are refused."""
import io

from pypdf import PdfWriter

from common.pdf import pdf_text


def make_pdf(*pages: list[str]) -> bytes:
    """A minimal PDF with one Helvetica text line per string; an empty page list is a blank page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        ops = "".join(f"BT /F1 11 Tf 50 {750 - 16 * i} Td ({line}) Tj ET\n" for i, line in enumerate(lines))
        objs.append(f"<< /Length {len(ops)} >>\nstream\n{ops}endstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


REPORT = ["SURGICAL PATHOLOGY REPORT", "Patient: Mary Tan  NRIC: S1234567D", "DIAGNOSIS: Invasive ductal carcinoma, 2.1 cm."]


def test_text_pdf_is_read_with_lf_line_endings():
    out = pdf_text(make_pdf(REPORT))
    assert "Mary Tan" in out["text"] and "2.1 cm" in out["text"] and "\r" not in out["text"]
    assert out["warnings"] == []


def test_scanned_page_is_skipped_with_a_warning():
    out = pdf_text(make_pdf(REPORT, []))
    assert "Invasive ductal carcinoma" in out["text"] and "Page 2 had no selectable text" in out["warnings"][0]


def test_scans_images_other_formats_and_locked_pdfs_are_refused():
    assert "looks scanned" in pdf_text(make_pdf([]))["error"]                       # no text layer
    assert "not a PDF" in pdf_text(b"\x89PNG\r\n\x1a\n" + b"0" * 100)["error"]        # an image renamed .pdf
    assert "could not be read" in pdf_text(b"%PDF-1.4\nnot really a pdf")["error"]
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt("secret")
    buf = io.BytesIO(); w.write(buf)
    assert "password-protected" in pdf_text(buf.getvalue())["error"]

