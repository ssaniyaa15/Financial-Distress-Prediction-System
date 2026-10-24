"""Pydantic schemas shared by services."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMRiskOutput(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    model_name: str = ""


class ModalityScore(BaseModel):
    probability: float
    risk_level: str
    details: dict[str, Any] = Field(default_factory=dict)


class RiskReport(BaseModel):
    company_id: str
    company_name: str = ""
    structured_risk: float
    text_risk: float
    vision_risk: float
    llm_risk: float
    early_fusion_risk: float | None = None
    intermediate_fusion_risk: float | None = None
    late_fusion_risk: float | None = None
    final_risk_score: float
    risk_level: str
    confidence: float
    contributing_factors: list[str] = Field(default_factory=list)
    shap_explanations: list[dict[str, Any]] = Field(default_factory=list)
    rule_based_factors: list[str] = Field(default_factory=list)
    explanation_source: str = "shap"
    decision_threshold: float | None = None
    structured_predicted_label: int | None = None
    llm_outputs: list[LLMRiskOutput] = Field(default_factory=list)
    ocr_text: str = ""
    extracted_fields: dict[str, Any] = Field(default_factory=dict)


class CompanyRecord(BaseModel):
    company_id: str
    features: dict[str, float]
    narrative: str = ""
    bankrupt: int | None = None
    split: str | None = None


class PredictRequest(BaseModel):
    company_id: str | None = None
    company_name: str = "Custom Company"
    features: dict[str, float] | None = None
    narrative: str | None = None
    fusion_method: str = "late"
