#!/usr/bin/env python3
"""Train a small CNN on financial-ratio charts generated from real features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import RocCurveDisplay
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.charts import chart_to_chw, render_ratio_chart
from shared.config import MODELS_DIR, PLOTS_DIR, VISION_MODEL_PATH, ensure_directories
from shared.dataset import labels, load_companies, split_frames
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.nn_models import RiskCNN

logger = get_logger("train_vision")
BATCH_SIZE = 64
EPOCHS = 8


class ChartDataset(Dataset):
    def __init__(self, frame) -> None:
        self.records = frame.to_dict(orient="records")
        self.targets = labels(frame)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image = chart_to_chw(render_ratio_chart(self.records[index]))
        target = int(self.targets[index])
        return torch.from_numpy(image), torch.tensor(target, dtype=torch.long)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict(model: RiskCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, preds = [], []
    for xb, _ in loader:
        logits = model(xb.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds.append((prob >= 0.5).astype(int))
        probs.append(prob)
    return np.concatenate(preds), np.concatenate(probs)


def main() -> None:
    ensure_directories()
    frame = load_companies()
    train, val, test = split_frames(frame)
    device = _device()
    train_loader = DataLoader(ChartDataset(train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ChartDataset(val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChartDataset(test), batch_size=BATCH_SIZE, shuffle=False)

    y_train = labels(train)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int((y_train == 0).sum())
    model = RiskCNN().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(xb)
        pred, proba = _predict(model, val_loader, device)
        metrics = classification_metrics(labels(val), pred, proba)
        logger.info(
            "epoch %s loss=%.4f val_auc=%.4f val_pr_auc=%.4f val_f1=%.4f",
            epoch,
            running / len(train),
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
        )
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state}, VISION_MODEL_PATH)
    pred, proba = _predict(model, test_loader, device)
    test_metrics = classification_metrics(labels(test), pred, proba)
    (MODELS_DIR / "vision_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels(test), proba, ax=ax, name="CNN")
    ax.set_title("Vision CNN ROC")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vision_evaluation.png", dpi=140)
    plt.close(fig)
    logger.info("Vision test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
