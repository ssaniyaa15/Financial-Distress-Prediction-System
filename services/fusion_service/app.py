from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FORCE_CPU", "1")

import json
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from shared.config import PROJECT_ROOT, ensure_directories
from shared.dataset import load_companies
from shared.json_utils import to_jsonable
from shared.logging_utils import get_logger
from shared.simple_inputs import company_snapshot, features_from_simple, narrative_from_simple
from shared.store import list_predictions
from shared.workflow import run_risk_workflow

logger = get_logger("fusion_service")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
ensure_directories()

app = FastAPI(title="Financial Risk Fusion Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fusion"}


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/companies")
def companies(limit: int = 50, mix: bool = True) -> dict:
    frame = load_companies()
    if mix:
        distressed = frame[frame["bankrupt"] == 1]
        healthy = frame[frame["bankrupt"] == 0]
        n_each = max(limit // 2, 1)
        subset = pd.concat(
            [
                distressed.head(n_each),
                healthy.head(n_each),
            ],
            ignore_index=True,
        )
    else:
        subset = frame.head(limit)
    items = []
    for row in subset.to_dict(orient="records"):
        snap = company_snapshot(row)
        items.append(
            {
                "company_id": row["company_id"],
                "company_name": row["company_name"],
                "label": "Distressed" if int(row["bankrupt"]) == 1 else "Healthy",
                "blurb": snap["blurb"],
            }
        )
    return to_jsonable({"count": int(len(frame)), "items": items})


@app.get("/companies/{company_id}")
def company(company_id: str) -> dict:
    frame = load_companies()
    hit = frame[frame["company_id"] == company_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail="Company not found")
    return to_jsonable(hit.iloc[0].to_dict())


@app.get("/history")
def history(limit: int = 25) -> dict:
    return {"items": list_predictions(limit)}


@app.post("/predict")
@app.post("/fusion")
async def predict(
    company_id: str | None = Form(default=None),
    company_name: str = Form(default="Custom Company"),
    features_json: str | None = Form(default=None),
    simple_json: str | None = Form(default=None),
    narrative: str | None = Form(default=None),
    pdf: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
) -> dict:
    state: dict = {
        "company_id": (company_id or "").strip() or "CUSTOM-00001",
        "company_name": company_name,
        "narrative": (narrative or "").strip(),
        "errors": [],
    }
    if features_json:
        try:
            parsed = json.loads(features_json)
            if parsed:
                state["features"] = parsed
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid features_json: {exc}") from exc
    elif simple_json:
        try:
            simple = json.loads(simple_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid simple_json: {exc}") from exc
        state["features"] = features_from_simple(simple)
        if not state["narrative"]:
            state["narrative"] = narrative_from_simple(simple, state["features"])

    temp_files: list[Path] = []
    try:
        if pdf is not None and pdf.filename:
            path = await _save_upload(pdf)
            temp_files.append(path)
            state["pdf_path"] = str(path)
        if image is not None and image.filename:
            path = await _save_upload(image)
            temp_files.append(path)
            state["image_path"] = str(path)
        result = run_risk_workflow(state)
        payload = result.get("report") or result
        return to_jsonable(payload)
    finally:
        for path in temp_files:
            path.unlink(missing_ok=True)


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.bin").suffix
    data = await upload.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return Path(handle.name)
