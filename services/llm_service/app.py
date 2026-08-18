from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from shared.llm import available_analyzers, ensemble_llm_outputs
from shared.logging_utils import get_logger

logger = get_logger("llm_service")
app = FastAPI(title="Financial Risk LLM Service", version="1.0.0")


class AnalyzeIn(BaseModel):
    text: str
    structured_summary: dict | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "llm",
        "analyzers": [item.name for item in available_analyzers()],
    }


@app.post("/analyze-text")
@app.post("/llm-analysis")
def analyze(payload: AnalyzeIn) -> dict:
    outputs = []
    for analyzer in available_analyzers():
        try:
            outputs.append(analyzer.analyze(payload.text, payload.structured_summary))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analyzer %s failed", analyzer.name)
            from shared.schemas import LLMRiskOutput

            outputs.append(
                LLMRiskOutput(
                    risk_score=0.5,
                    risk_level="medium",
                    confidence=0.2,
                    reasoning=str(exc),
                    model_name=analyzer.name,
                )
            )
    combined = ensemble_llm_outputs(outputs)
    return {
        "ensemble": combined.model_dump(),
        "models": [item.model_dump() for item in outputs],
    }
