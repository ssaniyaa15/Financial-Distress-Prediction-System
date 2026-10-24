from __future__ import annotations

import json

import numpy as np

from shared.config import DEFAULT_DECISION_THRESHOLD
from shared.threshold import load_decision_threshold, select_f1_threshold


def test_load_decision_threshold_missing_file_returns_half(tmp_path) -> None:
    missing = tmp_path / "structured_threshold.json"
    assert load_decision_threshold(missing) == DEFAULT_DECISION_THRESHOLD


def test_load_decision_threshold_reads_persisted_value(tmp_path) -> None:
    path = tmp_path / "structured_threshold.json"
    path.write_text(json.dumps({"threshold": 0.27, "metric": "f1"}), encoding="utf-8")
    assert load_decision_threshold(path) == 0.27


def test_load_decision_threshold_invalid_payload_falls_back(tmp_path) -> None:
    bad = tmp_path / "structured_threshold.json"
    bad.write_text("not-json", encoding="utf-8")
    assert load_decision_threshold(bad) == DEFAULT_DECISION_THRESHOLD
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"threshold": 1.5}), encoding="utf-8")
    assert load_decision_threshold(empty) == DEFAULT_DECISION_THRESHOLD


def test_select_f1_threshold_uses_only_provided_labels() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    proba = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.61, 0.91, 0.88])
    threshold, info = select_f1_threshold(y, proba)
    pred_half = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= threshold).astype(int)
    from sklearn.metrics import f1_score

    assert info["val_f1"] >= f1_score(y, pred_half, zero_division=0)
    assert f1_score(y, pred_tuned, zero_division=0) == info["val_f1"]
    assert 0.0 < threshold < 1.0
