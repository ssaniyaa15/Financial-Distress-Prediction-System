from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FORCE_CPU", "1")

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from shared.inference import ModelBundle, image_from_path
from shared.logging_utils import get_logger
from shared.metrics import risk_level_from_score

logger = get_logger("vision_service")
app = FastAPI(title="Financial Risk Vision Service", version="1.0.0")


class VisionJSON(BaseModel):
    features: dict[str, float]


@lru_cache(maxsize=1)
def bundle() -> ModelBundle:
    return ModelBundle()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vision"}


@app.post("/vision")
async def vision(
    file: UploadFile | None = File(default=None),
    features_json: str | None = Form(default=None),
) -> dict:
    models = bundle()
    features: dict[str, float] = {}
    if features_json:
        import json

        features = json.loads(features_json)
    image = None
    if file is not None:
        suffix = Path(file.filename or "chart.png").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(await file.read())
            path = Path(handle.name)
        try:
            image = image_from_path(path)
        finally:
            path.unlink(missing_ok=True)
    proba, embedding = models.vision_proba_and_embedding(features, image=image)
    return {
        "vision_risk": proba,
        "risk_level": risk_level_from_score(proba),
        "embedding_norm": float(np.linalg.norm(embedding)),
    }
