"""LangGraph node functions. Loaded via importlib to avoid shadowing the langgraph package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.dataset import load_companies
from shared.inference import ModelBundle, image_from_path
from shared.llm import ensemble_llm_outputs
from shared.metrics import risk_level_from_score
from shared.narratives import contributing_factors, generate_narrative, structured_summary
from shared.ocr import parse_financial_document
from shared.store import save_prediction

_BUNDLE: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = ModelBundle()
    return _BUNDLE


def load_data(state: dict[str, Any]) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    features = dict(state.get("features") or {})
    narrative = state.get("narrative") or ""
    company_id = state.get("company_id") or "CUSTOM-00001"
    company_name = state.get("company_name") or "Custom Company"
    company_id = state.get("company_id") or "CUSTOM-00001"
    if company_id.startswith("PL-"):
        frame = load_companies()
        hit = frame[frame["company_id"] == company_id]
        if hit.empty:
            errors.append(f"Unknown company_id {company_id}")
        else:
            row = hit.iloc[0].to_dict()
            company_name = row.get("company_name", company_name)
            features = row
            if not narrative:
                narrative = row.get("narrative") or generate_narrative(row)
    if features and not narrative:
        narrative = generate_narrative(features)
    return {
        "company_id": company_id,
        "company_name": company_name,
        "features": features,
        "narrative": narrative,
        "errors": errors,
    }


def ocr_analysis(state: dict[str, Any]) -> dict[str, Any]:
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        return {"ocr_text": state.get("narrative", ""), "extracted_fields": {}}
    parsed = parse_financial_document(Path(pdf_path))
    text = parsed["text"] or state.get("narrative", "")
    return {"ocr_text": text, "extracted_fields": parsed["extracted_fields"], "narrative": text or state.get("narrative", "")}


def structured_analysis(state: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    proba = bundle.structured_proba(state.get("features") or {})
    return {"structured_risk": proba}


def text_analysis(state: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    narrative = state.get("ocr_text") or state.get("narrative") or ""
    proba, _ = bundle.text_proba_and_embedding(narrative)
    return {"text_risk": proba}


def vision_analysis(state: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    features = state.get("features") or {}
    image = None
    if state.get("image_path"):
        image = image_from_path(Path(state["image_path"]))
    proba, _ = bundle.vision_proba_and_embedding(features, image=image)
    return {"vision_risk": proba}


def llm_analysis(state: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    narrative = state.get("ocr_text") or state.get("narrative") or ""
    outputs = bundle.llm_outputs(narrative, structured_summary(state.get("features") or {}))
    combined = ensemble_llm_outputs(outputs)
    return {
        "llm_risk": combined.risk_score,
        "llm_outputs": [item.model_dump() for item in outputs],
    }


def fusion(state: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    features = state.get("features") or {}
    narrative = state.get("ocr_text") or state.get("narrative") or ""
    image = image_from_path(Path(state["image_path"])) if state.get("image_path") else None
    s_vec = bundle.structured_vector(features)
    t_prob, t_emb = bundle.text_proba_and_embedding(narrative)
    v_prob, v_emb = bundle.vision_proba_and_embedding(features, image=image)
    fused = np.concatenate([s_vec, t_emb, v_emb])
    early = bundle.early_fusion_proba(fused)
    intermediate = bundle.intermediate_fusion_proba(s_vec, t_emb, v_emb)
    late_parts = np.array(
        [
            float(state.get("structured_risk") or 0.0),
            float(state.get("text_risk") or t_prob),
            float(state.get("vision_risk") or v_prob),
            float(state.get("llm_risk") or 0.0),
        ]
    )
    late = bundle.late_fusion_proba(late_parts)
    return {
        "text_risk": t_prob,
        "vision_risk": v_prob,
        "early_fusion_risk": early,
        "intermediate_fusion_risk": intermediate,
        "late_fusion_risk": late,
    }


def risk_prediction(state: dict[str, Any]) -> dict[str, Any]:
    scores = [
        state.get("late_fusion_risk"),
        state.get("intermediate_fusion_risk"),
        state.get("early_fusion_risk"),
        state.get("structured_risk"),
    ]
    final = next((float(score) for score in scores if score is not None), 0.5)
    parts = [
        state.get("structured_risk"),
        state.get("text_risk"),
        state.get("vision_risk"),
        state.get("llm_risk"),
    ]
    present = [float(p) for p in parts if p is not None]
    agreement = 1.0 - float(np.std(present)) if present else 0.5
    confidence = float(np.clip(0.45 + 0.5 * agreement, 0.05, 0.99))
    return {
        "final_risk_score": final,
        "risk_level": risk_level_from_score(final),
        "confidence": confidence,
        "contributing_factors": contributing_factors(state.get("features") or {}, final),
    }


def generate_report(state: dict[str, Any]) -> dict[str, Any]:
    report = {
        "company_id": state.get("company_id"),
        "company_name": state.get("company_name"),
        "structured_risk": state.get("structured_risk"),
        "text_risk": state.get("text_risk"),
        "vision_risk": state.get("vision_risk"),
        "llm_risk": state.get("llm_risk"),
        "early_fusion_risk": state.get("early_fusion_risk"),
        "intermediate_fusion_risk": state.get("intermediate_fusion_risk"),
        "late_fusion_risk": state.get("late_fusion_risk"),
        "final_risk_score": state.get("final_risk_score"),
        "risk_level": state.get("risk_level"),
        "confidence": state.get("confidence"),
        "contributing_factors": state.get("contributing_factors"),
        "llm_outputs": state.get("llm_outputs"),
        "ocr_text": (state.get("ocr_text") or "")[:4000],
        "extracted_fields": state.get("extracted_fields") or {},
        "errors": state.get("errors") or [],
    }
    try:
        save_prediction(report["company_id"], report.get("company_name") or "", report)
    except Exception:
        pass
    return {"report": report}
