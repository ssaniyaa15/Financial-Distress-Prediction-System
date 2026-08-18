"""OCR helpers: PDF/image to cleaned financial text."""

from __future__ import annotations

import re
from pathlib import Path

from shared.logging_utils import get_logger
from shared.narratives import extract_financial_fields

logger = get_logger("ocr")


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_image_path(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for image OCR") from exc
    with Image.open(path) as image:
        text = pytesseract.image_to_string(image)
    return _clean_text(text)


def extract_pdf_text(path: Path) -> str:
    """Prefer a PDF text layer; fall back to Tesseract OCR on rendered pages."""
    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypdf failed for %s: %s", path, exc)
    if len(_clean_text(text)) >= 80:
        return _clean_text(text)

    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("Tesseract/pdf2image unavailable; returning partial PDF text")
        return _clean_text(text)

    try:
        pages = convert_from_path(str(path), dpi=150)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf2image failed: %s", exc)
        return _clean_text(text)
    chunks = [pytesseract.image_to_string(page) for page in pages[:4]]
    return _clean_text("\n".join(chunks))


def parse_financial_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        text = ocr_image_path(path)
    else:
        text = extract_pdf_text(path)
    return {
        "text": text,
        "extracted_fields": extract_financial_fields(text),
        "source": str(path),
    }
