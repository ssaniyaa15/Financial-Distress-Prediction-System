"""Per-prediction SHAP explanations for the structured tabular model."""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.features import FEATURE_COLUMNS, FEATURE_DEFINITIONS

TREE_MODEL_NAMES = {"xgboost", "random_forest"}


def build_tree_explainer(model: Any):
    """Fit a TreeExplainer on an already-trained XGBoost or Random Forest model."""
    import shap

    return shap.TreeExplainer(model)


def positive_class_shap(raw: Any, n_features: int) -> np.ndarray:
    """Normalize TreeExplainer output to shape (n_samples, n_features) for class 1."""
    if isinstance(raw, list):
        arr = np.asarray(raw[1] if len(raw) > 1 else raw[0], dtype=np.float64)
        return arr
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, :, -1]
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape[-1] != n_features and arr.shape[0] == n_features:
        return arr.T
    return arr


def shap_matrix(explainer: Any, X: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(X)
    return positive_class_shap(values, X.shape[1])


def top_k_shap_explanations(
    shap_row: np.ndarray,
    feature_names: list[str] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k features by |SHAP| for a single row, with sign/direction."""
    names = feature_names or FEATURE_COLUMNS
    row = np.asarray(shap_row, dtype=np.float64).reshape(-1)
    k = min(k, row.size)
    order = np.argsort(np.abs(row))[::-1][:k]
    items: list[dict[str, Any]] = []
    for idx in order:
        value = float(row[idx])
        name = names[int(idx)]
        items.append(
            {
                "feature": name,
                "label": FEATURE_DEFINITIONS.get(name, name),
                "shap_value": round(value, 6),
                "abs_shap": round(abs(value), 6),
                "direction": "increases" if value > 0 else "decreases",
            }
        )
    return items


def format_shap_factor(item: dict[str, Any]) -> str:
    sign = (
        "increases distress risk"
        if item.get("direction") == "increases"
        else "decreases distress risk"
    )
    feature = item.get("feature", "?")
    label = item.get("label", feature)
    value = float(item.get("shap_value", 0.0))
    return f"{feature} ({label}): SHAP={value:+.4f} ({sign})"


def explain_structured_row(
    explainer: Any,
    X_row: np.ndarray,
    feature_names: list[str] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    X = np.asarray(X_row, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    matrix = shap_matrix(explainer, X)
    return top_k_shap_explanations(matrix[0], feature_names or FEATURE_COLUMNS, k=k)
