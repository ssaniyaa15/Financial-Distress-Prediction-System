#!/usr/bin/env python3
"""Train early, intermediate, and late multimodal fusion models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["FORCE_CPU"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fusion.early_fusion import stack_early_features
from fusion.late_fusion import fit_late_fusion
from shared.config import (
    EARLY_FUSION_PATH,
    INTERMEDIATE_FUSION_PATH,
    LATE_FUSION_PATH,
    MODELS_DIR,
    ensure_directories,
)
from shared.charts import chart_to_chw, render_ratio_chart
from shared.dataset import labels, load_companies, split_frames
from shared.inference import ModelBundle
from shared.llm import HeuristicFinancialAnalyzer
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.narratives import structured_summary
from shared.nn_models import IntermediateFusionNet
from shared.text_data import texts_to_sequences

logger = get_logger("train_fusion")


def collect_modalities(bundle: ModelBundle, frame, batch_size: int = 64) -> dict[str, np.ndarray]:
    structured = bundle.preprocessor.transform(frame)
    p_structured = bundle.structured_model.predict_proba(structured)[:, 1]

    sequences = texts_to_sequences(frame["narrative"].tolist(), bundle.vocab, bundle.lstm_max_len)
    text_embs: list[np.ndarray] = []
    text_probs: list[np.ndarray] = []
    bundle.lstm.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            tensor = torch.from_numpy(sequences[start : start + batch_size]).to(bundle.device)
            logits = bundle.lstm(tensor)
            text_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            text_embs.append(bundle.lstm.encode(tensor).cpu().numpy())

    records = frame.to_dict(orient="records")
    vis_embs: list[np.ndarray] = []
    vis_probs: list[np.ndarray] = []
    bundle.cnn.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            images = np.stack([chart_to_chw(render_ratio_chart(row)) for row in chunk])
            tensor = torch.from_numpy(images).to(bundle.device)
            logits = bundle.cnn(tensor)
            vis_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            vis_embs.append(bundle.cnn.encode(tensor).cpu().numpy())
            del tensor, images

    analyzer = HeuristicFinancialAnalyzer()
    p_llm = np.array(
        [analyzer.analyze(row["narrative"], structured_summary(row)).risk_score for row in records],
        dtype=np.float32,
    )
    return {
        "structured": structured.astype(np.float32),
        "text": np.vstack(text_embs),
        "vision": np.vstack(vis_embs),
        "p_structured": p_structured.astype(np.float32),
        "p_text": np.concatenate(text_probs),
        "p_vision": np.concatenate(vis_probs),
        "p_llm": p_llm,
    }


def train_intermediate(
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    val_pack: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> IntermediateFusionNet:
    device = bundle_device()
    model = IntermediateFusionNet(
        struct_dim=train_pack["structured"].shape[1],
        text_dim=train_pack["text"].shape[1],
        vision_dim=train_pack["vision"].shape[1],
    ).to(device)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int((y_train == 0).sum())
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def tensors(pack, y):
        return (
            torch.from_numpy(pack["structured"].astype(np.float32)),
            torch.from_numpy(pack["text"].astype(np.float32)),
            torch.from_numpy(pack["vision"].astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )

    train_ds = TensorDataset(*tensors(train_pack, y_train))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    best_state, best_auc = None, -1.0
    for epoch in range(1, 12):
        model.train()
        for xs, xt, xv, yb in loader:
            xs, xt, xv, yb = xs.to(device), xt.to(device), xv.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xs, xt, xv), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xs, xt, xv, _ = [t.to(device) for t in tensors(val_pack, y_val)]
            proba = torch.softmax(model(xs, xt, xv), dim=1)[:, 1].cpu().numpy()
        pred = (proba >= 0.5).astype(int)
        auc = classification_metrics(y_val, pred, proba)["roc_auc"]
        logger.info("intermediate epoch %s val_auc=%.4f", epoch, auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "struct_dim": train_pack["structured"].shape[1],
            "text_dim": train_pack["text"].shape[1],
            "vision_dim": train_pack["vision"].shape[1],
        },
        INTERMEDIATE_FUSION_PATH,
    )
    return model


def bundle_device() -> torch.device:
    return torch.device("cpu")


def main() -> None:
    ensure_directories()
    logger.info("Loading trained unimodal models on CPU for fusion training")
    os.environ["FORCE_CPU"] = "1"
    bundle = ModelBundle()
    frame = load_companies()
    train, val, test = split_frames(frame)
    logger.info("Extracting train modality features")
    train_pack = collect_modalities(bundle, train)
    logger.info("Extracting val modality features")
    val_pack = collect_modalities(bundle, val)
    logger.info("Extracting test modality features")
    test_pack = collect_modalities(bundle, test)
    y_train, y_val, y_test = labels(train), labels(val), labels(test)

    X_train = stack_early_features(train_pack["structured"], train_pack["text"], train_pack["vision"])
    X_val = stack_early_features(val_pack["structured"], val_pack["text"], val_pack["vision"])
    X_test = stack_early_features(test_pack["structured"], test_pack["text"], test_pack["vision"])
    early = LogisticRegression(max_iter=2000, class_weight="balanced")
    early.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(early, EARLY_FUSION_PATH)

    train_intermediate(train_pack, y_train, val_pack, y_val)

    late4_train = np.column_stack(
        [train_pack["p_structured"], train_pack["p_text"], train_pack["p_vision"], train_pack["p_llm"]]
    )
    late4_val = np.column_stack(
        [val_pack["p_structured"], val_pack["p_text"], val_pack["p_vision"], val_pack["p_llm"]]
    )
    late = fit_late_fusion(np.vstack([late4_train, late4_val]), np.concatenate([y_train, y_val]))
    late3_train = np.column_stack(
        [train_pack["p_structured"], train_pack["p_text"], train_pack["p_vision"]]
    )
    late3_val = np.column_stack(
        [val_pack["p_structured"], val_pack["p_text"], val_pack["p_vision"]]
    )
    late3 = LogisticRegression(max_iter=2000, class_weight="balanced")
    late3.fit(np.vstack([late3_train, late3_val]), np.concatenate([y_train, y_val]))
    late["model_3"] = late3
    joblib.dump(late, LATE_FUSION_PATH)

    # persist test packs so evaluation does not re-extract embeddings
    np.savez_compressed(
        MODELS_DIR / "fusion_test_cache.npz",
        structured=test_pack["structured"],
        text=test_pack["text"],
        vision=test_pack["vision"],
        p_structured=test_pack["p_structured"],
        p_text=test_pack["p_text"],
        p_vision=test_pack["p_vision"],
        p_llm=test_pack["p_llm"],
        y=y_test,
        early_X=X_test,
    )
    logger.info("Fusion models saved")


if __name__ == "__main__":
    main()
