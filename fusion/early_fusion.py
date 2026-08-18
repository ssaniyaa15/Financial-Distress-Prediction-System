"""Early fusion: concatenate modality features, then classify."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def stack_early_features(
    structured: np.ndarray,
    text_emb: np.ndarray,
    vision_emb: np.ndarray,
) -> np.ndarray:
    return np.concatenate([structured, text_emb, vision_emb], axis=1)


def fit_early_fusion(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    return model
