"""LangGraph workflow state."""

from typing import Any, TypedDict


class RiskState(TypedDict, total=False):
    company_id: str
    company_name: str
    features: dict[str, Any]
    narrative: str
    pdf_path: str
    image_path: str
    ocr_text: str
    extracted_fields: dict[str, Any]
    structured_risk: float
    text_risk: float
    vision_risk: float
    llm_risk: float
    llm_outputs: list[dict[str, Any]]
    early_fusion_risk: float
    intermediate_fusion_risk: float
    late_fusion_risk: float
    final_risk_score: float
    risk_level: str
    confidence: float
    contributing_factors: list[str]
    shap_explanations: list[dict[str, Any]]
    rule_based_factors: list[str]
    explanation_source: str
    decision_threshold: float
    structured_predicted_label: int
    report: dict[str, Any]
    errors: list[str]
