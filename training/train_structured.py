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
    SHAP_EXPLAINER_PATH,
    SHAP_TEST_VALUES_PATH,
    STRUCTURED_METRICS_PATH,
    STRUCTURED_MODEL_PATH,
    ensure_directories,
)
from shared.dataset import labels, load_companies, split_frames
from shared.explain import TREE_MODEL_NAMES, build_tree_explainer, shap_matrix
from shared.features import FEATURE_COLUMNS
from shared.json_utils import to_jsonable
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.preprocessing import fit_preprocessor
from shared.threshold import save_decision_threshold, select_f1_threshold

logger = get_logger("train_structured")


def _eval(model, X, y, threshold: float = 0.5) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
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
        logger.info(
            "%s val ROC-AUC=%.4f val PR-AUC=%.4f test F1=%.4f",
            name,
            val_metrics["roc_auc"],
            val_metrics["pr_auc"],
            test_metrics["f1"],
        )
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    val_proba = best_model.predict_proba(X_val)[:, 1]
    threshold, threshold_info = select_f1_threshold(y_val, val_proba)
    save_decision_threshold(threshold, extra=threshold_info)
    test_at_half = _eval(best_model, X_test, y_test, threshold=0.5)
    test_at_tuned = _eval(best_model, X_test, y_test, threshold=threshold)
    logger.info(
        "Val-tuned F1 threshold=%.3f (val F1=%.4f). Test F1@0.5=%.4f vs F1@tuned=%.4f",
        threshold,
        threshold_info["val_f1"],
        test_at_half["f1"],
        test_at_tuned["f1"],
    )
    logger.info("Test@0.5 %s", test_at_half)
    logger.info("Test@tuned %s", test_at_tuned)

    joblib.dump(
        {"model": best_model, "name": best_name, "threshold": threshold},
        STRUCTURED_MODEL_PATH,
    )
    report["best_model"] = best_name
    report["decision_threshold"] = {
        "threshold": threshold,
        "selected_on": "val",
        "metric": "f1",
        "val_f1": threshold_info["val_f1"],
        "test_at_0.5": test_at_half,
        "test_at_tuned_threshold": test_at_tuned,
    }

    if best_name in TREE_MODEL_NAMES:
        logger.info("Computing TreeExplainer SHAP values for %s", best_name)
        explainer = build_tree_explainer(best_model)
        test_shap = shap_matrix(explainer, X_test)
        joblib.dump(
            {
                "explainer": explainer,
                "feature_names": FEATURE_COLUMNS,
                "model_name": best_name,
            },
            SHAP_EXPLAINER_PATH,
        )
        expected = getattr(explainer, "expected_value", None)
        if expected is not None:
            expected = np.asarray(expected, dtype=np.float64)
        np.savez_compressed(
            SHAP_TEST_VALUES_PATH,
            values=test_shap,
            expected_value=expected if expected is not None else np.array([]),
            feature_names=np.array(FEATURE_COLUMNS),
        )
        report["shap"] = {
            "explainer_path": str(SHAP_EXPLAINER_PATH),
            "test_values_path": str(SHAP_TEST_VALUES_PATH),
            "n_test": int(test_shap.shape[0]),
        }
        logger.info("Saved SHAP explainer and test values (%s rows)", test_shap.shape[0])
    else:
        logger.warning("Best model %s is not tree-based; skipping TreeExplainer", best_name)

    STRUCTURED_METRICS_PATH.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name=best_name)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], colorbar=False)
    axes[0].set_title("Structured model ROC")
    axes[1].set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "structured_evaluation.png", dpi=140)
    plt.close(fig)
    logger.info("Saved best structured model (%s) to %s", best_name, STRUCTURED_MODEL_PATH)


if __name__ == "__main__":
    main()
