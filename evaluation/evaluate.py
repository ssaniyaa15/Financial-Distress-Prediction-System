#!/usr/bin/env python3
"""Evaluate every trained approach on the held-out test set. No fabricated metrics."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["FORCE_CPU"] = "1"

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import (
    EARLY_FUSION_PATH,
    EVAL_RESULTS_PATH,
    INTERMEDIATE_FUSION_PATH,
    LATE_FUSION_PATH,
    MODELS_DIR,
    PLOTS_DIR,
    STRUCTURED_METRICS_PATH,
    ensure_directories,
)
from shared.dataset import labels, load_companies, split_frames
from shared.inference import ModelBundle
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.nn_models import IntermediateFusionNet

logger = get_logger("evaluate")


def _pack_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(y, pred, proba)


def main() -> None:
    ensure_directories()
    cache_path = MODELS_DIR / "fusion_test_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError("Run training/train_fusion.py before evaluation.")
    cache = np.load(cache_path)
    y = cache["y"]
    bundle = ModelBundle()

    results: dict[str, dict[str, float]] = {}
    curves: dict[str, np.ndarray] = {}

    results["Structured Only"] = _pack_metrics(y, cache["p_structured"])
    curves["Structured Only"] = cache["p_structured"]
    results["Text Only"] = _pack_metrics(y, cache["p_text"])
    curves["Text Only"] = cache["p_text"]
    results["Vision Only"] = _pack_metrics(y, cache["p_vision"])
    curves["Vision Only"] = cache["p_vision"]

    early_proba = bundle.early.predict_proba(cache["early_X"])[:, 1]
    results["Early Fusion"] = _pack_metrics(y, early_proba)
    curves["Early Fusion"] = early_proba

    device = bundle.device
    inter = bundle.intermediate
    assert inter is not None
    with torch.no_grad():
        logits = inter(
            torch.from_numpy(cache["structured"].astype(np.float32)).to(device),
            torch.from_numpy(cache["text"].astype(np.float32)).to(device),
            torch.from_numpy(cache["vision"].astype(np.float32)).to(device),
        )
        inter_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    results["Intermediate Fusion"] = _pack_metrics(y, inter_proba)
    curves["Intermediate Fusion"] = inter_proba

    late3_X = np.column_stack([cache["p_structured"], cache["p_text"], cache["p_vision"]])
    # The saved late model was trained on 4 columns. Evaluate 3-modality late fusion
    # by fitting on the cached test predictions' complement... we persist a 3-way
    # model during train_fusion. If missing, derive it from the 4-way inputs
    # using the same test labels would leak. Use the saved 3-way model if present.
    late_pack = bundle.late
    if late_pack is not None and "model_3" in late_pack:
        late3_proba = late_pack["model_3"].predict_proba(late3_X)[:, 1]
    else:
        late3_proba = late3_X.mean(axis=1)
    results["Late Fusion"] = _pack_metrics(y, late3_proba)
    curves["Late Fusion"] = late3_proba

    late4_X = np.column_stack(
        [cache["p_structured"], cache["p_text"], cache["p_vision"], cache["p_llm"]]
    )
    late4_proba = late_pack["model"].predict_proba(late4_X)[:, 1]
    results["Multimodal + LLM"] = _pack_metrics(y, late4_proba)
    curves["Multimodal + LLM"] = late4_proba

    EVAL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    header = "| Approach | Accuracy | Precision | Recall | F1 | ROC-AUC |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    order = [
        "Structured Only",
        "Text Only",
        "Vision Only",
        "Early Fusion",
        "Intermediate Fusion",
        "Late Fusion",
        "Multimodal + LLM",
    ]
    for name in order:
        m = results[name]
        rows.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )
    markdown = "\n".join([header, sep, *rows])
    table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    (EVAL_RESULTS_PATH.parent / "comparison.md").write_text(
        "# Held-out test comparison\n\n" + markdown + "\n",
        encoding="utf-8",
    )
    print(markdown)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in curves.items():
        RocCurveDisplay.from_predictions(y, proba, ax=ax, name=name)
    ax.set_title("Held-out ROC by approach")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fusion_comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    table["roc_auc"].plot(kind="bar", ax=ax, color="#1f4e79")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Test ROC-AUC by approach")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "auc_bars.png", dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", EVAL_RESULTS_PATH)


if __name__ == "__main__":
    main()
