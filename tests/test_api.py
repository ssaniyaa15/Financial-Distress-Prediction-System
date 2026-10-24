from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient


REQUIRED_PREDICT_KEYS = {
    "company_id",
    "company_name",
    "final_risk_score",
    "risk_level",
    "confidence",
    "contributing_factors",
    "shap_explanations",
    "rule_based_factors",
    "explanation_source",
    "decision_threshold",
    "structured_predicted_label",
    "structured_risk",
}


def test_fusion_predict_endpoint_returns_expected_shape() -> None:
    from services.fusion_service.app import app

    report = {
        "company_id": "CUSTOM-00001",
        "company_name": "Acme",
        "structured_risk": 0.71,
        "text_risk": 0.22,
        "vision_risk": 0.31,
        "llm_risk": 0.40,
        "early_fusion_risk": 0.50,
        "intermediate_fusion_risk": 0.48,
        "late_fusion_risk": 0.66,
        "final_risk_score": 0.66,
        "risk_level": "high",
        "confidence": 0.82,
        "contributing_factors": [
            "Attr2 (total liabilities / total assets): SHAP=+0.1200 (increases distress risk)"
        ],
        "shap_explanations": [
            {
                "feature": "Attr2",
                "label": "total liabilities / total assets",
                "shap_value": 0.12,
                "abs_shap": 0.12,
                "direction": "increases",
            }
        ],
        "rule_based_factors": ["High leverage (liabilities / assets): total liabilities / total assets=0.8100"],
        "explanation_source": "shap",
        "decision_threshold": 0.31,
        "structured_predicted_label": 1,
        "llm_outputs": [],
        "extracted_fields": {},
        "errors": [],
    }
    with patch("services.fusion_service.app.run_risk_workflow", return_value={"report": report}):
        client = TestClient(app)
        response = client.post(
            "/predict",
            data={
                "company_name": "Acme",
                "simple_json": json.dumps(
                    {"profit": 0.04, "debt": 0.55, "current_ratio": 1.2, "equity": 0.4}
                ),
            },
        )
    assert response.status_code == 200
    body = response.json()
    missing = REQUIRED_PREDICT_KEYS - set(body)
    assert not missing, f"missing keys: {missing}"
    assert isinstance(body["shap_explanations"], list)
    assert body["shap_explanations"][0]["feature"] == "Attr2"
    assert body["explanation_source"] == "shap"
    assert 0.0 < body["decision_threshold"] < 1.0
