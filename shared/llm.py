"""Multiple-LLM financial risk analysis with a shared interface.

LLMs are called over HTTP APIs. No model weights are downloaded.
If no API keys are configured, a transparent heuristic analyzer is used so
training and local demos still run.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from shared.logging_utils import get_logger
from shared.metrics import risk_level_from_score
from shared.narratives import _num, extract_financial_fields
from shared.schemas import LLMRiskOutput

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a financial-distress analyst.
Score bankruptcy/distress risk from the provided financial discussion and ratios.
Return ONLY valid JSON with keys:
risk_score (float 0-1), risk_level (low|medium|high), confidence (float 0-1), reasoning (short string).
Do not invent financial figures that are not in the input.
"""


def _parse_llm_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object in LLM response")
    payload = json.loads(match.group(0))
    score = float(payload.get("risk_score", 0.5))
    score = min(max(score, 0.0), 1.0)
    confidence = float(payload.get("confidence", 0.5))
    confidence = min(max(confidence, 0.0), 1.0)
    level = str(payload.get("risk_level", risk_level_from_score(score))).lower()
    if level not in {"low", "medium", "high"}:
        level = risk_level_from_score(score)
    return {
        "risk_score": score,
        "risk_level": level,
        "confidence": confidence,
        "reasoning": str(payload.get("reasoning", ""))[:800],
    }


class FinancialLLMAnalyzer(ABC):
    name: str

    @abstractmethod
    def analyze(self, text: str, structured_summary: dict[str, Any] | None = None) -> LLMRiskOutput:
        raise NotImplementedError


class HeuristicFinancialAnalyzer(FinancialLLMAnalyzer):
    """Deterministic fallback used when API keys are absent.

    This is not an LLM. It applies transparent ratio/keyword rules so the
    pipeline can be evaluated without calling external providers.
    """

    name = "heuristic-rules"

    def analyze(self, text: str, structured_summary: dict[str, Any] | None = None) -> LLMRiskOutput:
        summary = structured_summary or {}
        fields = extract_financial_fields(text)
        leverage = _num(summary.get("Leverage", fields.get("debt", 0.5)))
        roa = _num(summary.get("ROA", fields.get("profit", 0.0)))
        current = _num(summary.get("CurrentRatio", 1.2))
        equity = _num(summary.get("EquityRatio", 0.4))
        margin = _num(summary.get("ProfitMargin", 0.0))
        wc = _num(summary.get("WorkingCapRatio", 0.1))

        score = 0.15
        reasons: list[str] = []
        if leverage > 0.7:
            score += 0.22
            reasons.append("leverage above 0.70")
        elif leverage > 0.5:
            score += 0.10
            reasons.append("moderate leverage")
        if roa < 0:
            score += 0.20
            reasons.append("negative ROA")
        elif roa < 0.03:
            score += 0.10
            reasons.append("low ROA")
        if current < 1:
            score += 0.14
            reasons.append("current ratio below 1")
        if equity < 0.2:
            score += 0.14
            reasons.append("thin equity buffer")
        if margin < 0:
            score += 0.10
            reasons.append("negative profit margin")
        if wc < 0:
            score += 0.08
            reasons.append("negative working capital")
        lowered = text.lower()
        if "elevated leverage" in lowered or "liquidity is tight" in lowered:
            score += 0.05
            reasons.append("cautious language in the discussion")
        score = min(max(score, 0.02), 0.98)
        reasoning = (
            "Heuristic financial-text scorer (no LLM API key). Signals: "
            + (", ".join(reasons) if reasons else "no major red flags")
            + "."
        )
        return LLMRiskOutput(
            risk_score=float(score),
            risk_level=risk_level_from_score(score),
            confidence=0.55 if reasons else 0.45,
            reasoning=reasoning,
            model_name=self.name,
        )


class OpenAIAnalyzer(FinancialLLMAnalyzer):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def analyze(self, text: str, structured_summary: dict[str, Any] | None = None) -> LLMRiskOutput:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"text": text[:6000], "ratios": structured_summary or {}}),
                },
            ],
        }
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=45.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content)
        return LLMRiskOutput(**parsed, model_name=f"openai:{self.model}")


class GeminiAnalyzer(FinancialLLMAnalyzer):
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def analyze(self, text: str, structured_summary: dict[str, Any] | None = None) -> LLMRiskOutput:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": SYSTEM_PROMPT
                            + "\n"
                            + json.dumps({"text": text[:6000], "ratios": structured_summary or {}})
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        response = httpx.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=45.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:400]}")
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _parse_llm_json(content)
        return LLMRiskOutput(**parsed, model_name=f"gemini:{self.model}")


class LlamaAnalyzer(FinancialLLMAnalyzer):
    """Llama via Groq or an OpenAI-compatible local/Ollama endpoint. Weights are not downloaded."""

    name = "llama"

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/")

    def analyze(self, text: str, structured_summary: dict[str, Any] | None = None) -> LLMRiskOutput:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"text": text[:6000], "ratios": structured_summary or {}}),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content)
        return LLMRiskOutput(**parsed, model_name=f"llama:{self.model}")


def available_analyzers() -> list[FinancialLLMAnalyzer]:
    analyzers: list[FinancialLLMAnalyzer] = []
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    ollama = os.getenv("OLLAMA_BASE_URL", "")
    if openai_key:
        analyzers.append(OpenAIAnalyzer(openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    if gemini_key:
        analyzers.append(GeminiAnalyzer(gemini_key, os.getenv("GEMINI_MODEL", "gemini-flash-latest")))
    if groq_key:
        analyzers.append(
            LlamaAnalyzer(groq_key, os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant"))
        )
    elif ollama:
        analyzers.append(
            LlamaAnalyzer(
                api_key="",
                model=os.getenv("LLAMA_MODEL", "llama3.1"),
                base_url=f"{ollama.rstrip('/')}/v1",
            )
        )
    if not analyzers:
        logger.warning("No LLM API keys found; using heuristic financial analyzer.")
        analyzers.append(HeuristicFinancialAnalyzer())
    return analyzers


def ensemble_llm_outputs(outputs: list[LLMRiskOutput]) -> LLMRiskOutput:
    if not outputs:
        return HeuristicFinancialAnalyzer().analyze("")
    score = float(sum(item.risk_score for item in outputs) / len(outputs))
    confidence = float(sum(item.confidence for item in outputs) / len(outputs))
    reasoning = " | ".join(f"{item.model_name}: {item.reasoning}" for item in outputs)
    return LLMRiskOutput(
        risk_score=score,
        risk_level=risk_level_from_score(score),
        confidence=confidence,
        reasoning=reasoning[:1200],
        model_name="ensemble:" + ",".join(item.model_name for item in outputs),
    )
