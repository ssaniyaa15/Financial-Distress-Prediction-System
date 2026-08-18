#!/usr/bin/env python3
"""Train logistic regression, random forest, and XGBoost on structured ratios."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import (
    MODELS_DIR,
    PLOTS_DIR,
    PREPROCESSOR_PATH,
    STRUCTURED_METRICS_PATH,
    STRUCTURED_MODEL_PATH,
    ensure_directories,
)
from shared.dataset import labels, load_companies, split_frames
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.preprocessing import fit_preprocessor

logger = get_logger("train_structured")


def _eval(model, X, y) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(y, pred, proba)


def main() -> None:
    ensure_directories()
    frame = load_companies()
    train, val, test = split_frames(frame)
    preprocessor = fit_preprocessor(train)
    preprocessor.save(PREPROCESSOR_PATH)

    X_train = preprocessor.transform(train)
    X_val = preprocessor.transform(val)
    X_test = preprocessor.transform(test)
    y_train, y_val, y_test = labels(train), labels(val), labels(test)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }

    report: dict[str, dict] = {}
    best_name = None
    best_auc = -1.0
    best_model = None
    for name, model in candidates.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        val_metrics = _eval(model, X_val, y_val)
        test_metrics = _eval(model, X_test, y_test)
        report[name] = {"val": val_metrics, "test": test_metrics}
        logger.info("%s val ROC-AUC=%.4f test F1=%.4f", name, val_metrics["roc_auc"], test_metrics["f1"])
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None
    joblib.dump({"model": best_model, "name": best_name}, STRUCTURED_MODEL_PATH)
    report["best_model"] = best_name
    STRUCTURED_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name=best_name)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], colorbar=False)
    axes[0].set_title("Structured model ROC")
    axes[1].set_title("Structured model confusion matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "structured_evaluation.png", dpi=140)
    plt.close(fig)
    logger.info("Saved best structured model (%s) to %s", best_name, STRUCTURED_MODEL_PATH)


if __name__ == "__main__":
    main()
