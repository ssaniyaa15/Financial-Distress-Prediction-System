from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FORCE_CPU", "1")

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from shared.explain import format_shap_factor
from shared.inference import ModelBundle
from shared.logging_utils import get_logger
from shared.metrics import risk_level_from_score
from shared.narratives import generate_narrative

logger = get_logger("ml_service")
app = FastAPI(title="Financial Risk ML Service", version="1.0.0")


class PredictIn(BaseModel):
    features: dict[str, float]
    narrative: str | None = None


@lru_cache(maxsize=1)
def bundle() -> ModelBundle:
    return ModelBundle()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ml"}


@app.post("/predict")
def predict(payload: PredictIn) -> dict:
    models = bundle()
    narrative = payload.narrative or generate_narrative(payload.features)
    structured = models.structured_proba(payload.features)
    text, _ = models.text_proba_and_embedding(narrative)
    shap_items = models.shap_explanations(payload.features, k=5)
    threshold = models.decision_threshold
    return {
        "structured_risk": structured,
        "text_risk": text,
        "structured_level": risk_level_from_score(structured),
        "text_level": risk_level_from_score(text),
        "structured_model": models.structured_name,
        "decision_threshold": threshold,
        "structured_predicted_label": int(structured >= threshold),
        "shap_explanations": shap_items,
        "contributing_factors": [format_shap_factor(item) for item in shap_items],
        "explanation_source": "shap" if shap_items else "unavailable",
    }
