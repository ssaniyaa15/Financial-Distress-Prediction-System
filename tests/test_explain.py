from __future__ import annotations

import numpy as np

from shared.explain import format_shap_factor, top_k_shap_explanations
from shared.features import FEATURE_COLUMNS, FEATURE_DEFINITIONS


def test_shap_explanation_output_has_expected_keys_and_top5() -> None:
    row = np.zeros(len(FEATURE_COLUMNS))
    row[1] = 0.40
    row[0] = -0.25
    row[9] = 0.11
    row[3] = -0.09
    row[50] = 0.07
    items = top_k_shap_explanations(row, FEATURE_COLUMNS, k=5)
    assert len(items) == 5
    required = {"feature", "label", "shap_value", "abs_shap", "direction"}
    for item in items:
        assert required <= set(item.keys())
        assert item["label"] == FEATURE_DEFINITIONS[item["feature"]]
        assert item["abs_shap"] == abs(item["shap_value"])
        assert item["direction"] in {"increases", "decreases"}
    assert items[0]["feature"] == "Attr2"
    assert items[0]["direction"] == "increases"
    assert items[1]["feature"] == "Attr1"
    assert items[1]["direction"] == "decreases"
    rendered = format_shap_factor(items[0])
    assert "Attr2" in rendered
    assert "SHAP=" in rendered
    assert "increases distress risk" in rendered
