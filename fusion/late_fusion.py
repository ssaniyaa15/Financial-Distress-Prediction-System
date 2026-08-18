"""Late fusion: combine per-modality probabilities with a stacking classifier."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_late_fusion(probabilities: np.ndarray, y: np.ndarray) -> dict:
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(probabilities, y)
    weights = {
        "structured": float(model.coef_[0][0]),
        "text": float(model.coef_[0][1]),
        "vision": float(model.coef_[0][2]),
        "llm": float(model.coef_[0][3]),
        "intercept": float(model.intercept_[0]),
    }
    return {"model": model, "weights": weights}


def weighted_average(probabilities: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(probabilities.shape[1], dtype=np.float64)
        weights = weights / weights.sum()
    return probabilities @ weights
