from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from shared.logging_utils import get_logger
from shared.ocr import parse_financial_document

logger = get_logger("ocr_service")
app = FastAPI(title="Financial Risk OCR Service", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ocr"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "document.pdf").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        temp_path = Path(handle.name)
    try:
        parsed = parse_financial_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return parsed
