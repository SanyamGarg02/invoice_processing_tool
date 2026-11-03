"""OCR for scanned / image-based pages using Tesseract (via pytesseract)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import pymupdf
from PIL import Image, ImageOps

from services.layout import WordBox, layout
from utils.helpers import get_logger

log = get_logger(__name__)

OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_LANG = os.getenv("OCR_LANG", "eng")


class OCRUnavailableError(RuntimeError):
    pass


@dataclass
class OCRResult:
    text: str
    rows: list
    mean_confidence: float | None
    word_count: int
    low_conf_ratio: float  # share of words with confidence < 60


def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_page(page: pymupdf.Page, dpi: int = OCR_DPI, clip: pymupdf.Rect | None = None) -> OCRResult:
    """Render a PDF page (or a region of it) to an image and OCR it.

    Word boxes are mapped back to PDF points so OCR rows and digital rows share
    one coordinate system.
    """
    import pytesseract

    if not ocr_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed. Install it with 'brew install tesseract' to read scanned PDFs."
        )
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    scale = 72.0 / dpi
    ox, oy = (clip.x0, clip.y0) if clip is not None else (0.0, 0.0)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = ImageOps.grayscale(img)
    # psm 6 = treat page as uniform blocks of text; works well for invoice tables.
    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    words: list[WordBox] = []
    confs: list[float] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not txt or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(WordBox(ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale, txt, conf))
        confs.append(conf)
    mean_conf = sum(confs) / len(confs) if confs else None
    low_ratio = (sum(1 for c in confs if c < 60) / len(confs)) if confs else 1.0
    log.info("OCR page %s: %d words, mean conf %s", page.number + 1, len(words), mean_conf and round(mean_conf, 1))
    text, rows = layout(words)
    return OCRResult(text, rows, mean_conf, len(words), low_ratio)
