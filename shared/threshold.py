"""Decision-threshold selection and loading for the structured model.

The default 0.5 cutoff is a poor operating point when the positive class is
~5%. Thresholds are chosen on the validation split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from shared.config import DEFAULT_DECISION_THRESHOLD, DECISION_THRESHOLD_PATH


def select_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the probability cutoff that maximizes F1 on the provided labels.

    Call this with validation labels only. Never pass the test set.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_f1 = -1.0
    for threshold in grid:
        pred = (y_proba >= threshold).astype(int)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, {"threshold": best_threshold, "val_f1": best_f1}


def save_decision_threshold(
    threshold: float,
    path: Path | None = None,
    extra: dict | None = None,
) -> Path:
    destination = path or DECISION_THRESHOLD_PATH
    payload = {"threshold": float(threshold), "metric": "f1", "split": "val"}
    if extra:
        payload.update(extra)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_decision_threshold(path: Path | None = None) -> float:
    """Return the persisted cutoff, or 0.5 if the file is missing/invalid."""
    source = path if path is not None else DECISION_THRESHOLD_PATH
    if not source.exists():
        return DEFAULT_DECISION_THRESHOLD
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        value = float(payload["threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_DECISION_THRESHOLD
    if not (0.0 < value < 1.0):
        return DEFAULT_DECISION_THRESHOLD
    return value
