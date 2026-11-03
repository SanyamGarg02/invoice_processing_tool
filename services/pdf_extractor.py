"""PDF validation and page-wise text extraction with OCR fallback.

For every page we decide independently whether its digital text layer is usable
or whether the page is a scanned image that needs OCR. This handles mixed PDFs
(some pages digital, some scanned) and avoids running OCR when not needed.
On digital pages, embedded images large enough to hold text (logos, letterheads
with company details) are OCR'd separately, because vendors often put their own
name only inside a logo image.
"""

from __future__ import annotations

import os
import re

import pymupdf

from models.invoice_schema import DocumentContent, PageContent
from services.layout import WordBox, layout
from services.ocr_service import OCRUnavailableError, ocr_available, ocr_page
from services.table_extractor import extract_tables
from utils.helpers import get_logger

log = get_logger(__name__)

MIN_DIGITAL_CHARS = 40  # below this a page is treated as having no real text layer
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
OCR_IMAGE_REGIONS = os.getenv("OCR_IMAGE_REGIONS", "true").lower() == "true"
LOW_OCR_CONFIDENCE = 70.0


class PDFValidationError(Exception):
    """Raised for files that cannot be processed; message is user-friendly."""


def open_pdf(data: bytes, filename: str) -> pymupdf.Document:
    if not data:
        raise PDFValidationError("The file is empty (0 bytes).")
    if b"%PDF" not in data[:1024]:
        raise PDFValidationError("The file is not a valid PDF document.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise PDFValidationError("The PDF is corrupted or uses an unsupported encoding and could not be opened.")
    if doc.needs_pass:
        doc.close()
        raise PDFValidationError("The PDF is password-protected. Please upload an unprotected copy.")
    if doc.page_count == 0:
        doc.close()
        raise PDFValidationError("The PDF contains no pages.")
    if doc.page_count > MAX_PAGES:
        n = doc.page_count
        doc.close()
        raise PDFValidationError(f"The PDF has {n} pages; the limit is {MAX_PAGES}.")
    return doc


def _text_is_garbage(text: str) -> bool:
    """Detect broken font encodings (text layer full of replacement/control chars)."""
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return False
    bad = sum(1 for c in stripped if c == "�" or ord(c) < 32)
    letters = len(re.findall(r"[^\W_]", stripped))
    return bad / len(stripped) > 0.1 or letters / len(stripped) < 0.4


def _image_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for img in page.get_images(full=True):
        try:
            rects.extend(r & page.rect for r in page.get_image_rects(img[0]))
        except Exception:
            continue
    return [r for r in rects if not r.is_empty]


def _image_coverage(page: pymupdf.Page) -> float:
    area = abs(page.rect)
    return min(sum(abs(r) for r in _image_rects(page)) / area, 1.0) if area else 0.0


def _ocr_image_regions(page: pymupdf.Page) -> str:
    """OCR embedded images on a digital page that are big enough to contain text."""
    area = abs(page.rect)
    texts = []
    for r in _image_rects(page)[:6]:
        if not (0.01 <= abs(r) / area <= 0.6) or r.height < 15:
            continue
        try:
            res = ocr_page(page, clip=r)
        except Exception as exc:
            log.debug("image-region OCR skipped: %s", exc)
            continue
        if res.word_count >= 2 and (res.mean_confidence or 0) >= 60:
            texts.append(res.text)
    return "\n".join(texts)


def extract_document(data: bytes, filename: str, use_tables: bool = True) -> DocumentContent:
    """Validate a PDF and return layout-preserving text for each page."""
    doc = open_pdf(data, filename)
    pages: list[PageContent] = []
    warnings: list[str] = []
    digital_pages: list[int] = []
    can_ocr = ocr_available()
    try:
        for page in doc:
            pno = page.number + 1
            try:
                words = [WordBox(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
                text, rows = layout(words)
            except Exception as exc:
                log.warning("%s p%d: digital extraction failed: %s", filename, pno, exc)
                text, rows = "", []
            n_chars = len(re.sub(r"\s", "", text))
            garbage = _text_is_garbage(text)
            needs_ocr = n_chars < MIN_DIGITAL_CHARS or garbage

            if needs_ocr and (_image_coverage(page) > 0.05 or n_chars == 0 or garbage):
                if not can_ocr:
                    warnings.append(f"Page {pno} looks scanned but Tesseract OCR is not installed.")
                    pages.append(PageContent(page_number=pno, text="" if garbage else text, method="empty", rows=rows))
                    continue
                try:
                    res = ocr_page(page)
                    if res.mean_confidence is not None and res.mean_confidence < LOW_OCR_CONFIDENCE:
                        warnings.append(f"Page {pno}: low OCR quality (mean confidence {res.mean_confidence:.0f}%).")
                    keep_digital = n_chars > 0 and not garbage
                    pages.append(PageContent(
                        page_number=pno,
                        text=(text + "\n" + res.text) if keep_digital else res.text,
                        method="ocr" if res.text.strip() else "empty",
                        ocr_confidence=res.mean_confidence,
                        rows=(rows + res.rows) if keep_digital else res.rows,
                    ))
                except OCRUnavailableError as exc:
                    warnings.append(f"Page {pno}: {exc}")
                    pages.append(PageContent(page_number=pno, text=text, method="empty", rows=rows))
                except Exception as exc:
                    log.warning("%s p%d: OCR failed: %s", filename, pno, exc)
                    warnings.append(f"Page {pno}: OCR failed for this page.")
                    pages.append(PageContent(page_number=pno, text=text, method="empty", rows=rows))
            else:
                if n_chars and can_ocr and OCR_IMAGE_REGIONS:
                    img_text = _ocr_image_regions(page)
                    if img_text:
                        text += "\n[Text recognised inside images on this page]\n" + img_text
                pages.append(PageContent(page_number=pno, text=text, method="digital" if n_chars else "empty", rows=rows))
                if n_chars:
                    digital_pages.append(pno)
    finally:
        doc.close()

    if use_tables and digital_pages:
        tables = extract_tables(data, digital_pages)
        for p in pages:
            p.tables_markdown = tables.get(p.page_number, [])

    log.info("%s: %d pages (%s)", filename, len(pages), ", ".join(p.method for p in pages))
    return DocumentContent(filename=filename, page_count=len(pages), pages=pages, warnings=warnings)


def render_page_png(data: bytes, page_number: int, dpi: int = 110) -> bytes | None:
    """Render one page as PNG for showing the original invoice to a reviewer."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return doc[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
    except Exception:
        return None
