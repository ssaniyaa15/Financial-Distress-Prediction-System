#!/usr/bin/env python3
"""Train an LSTM text classifier and a TF-IDF logistic baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import LSTM_MODEL_PATH, LSTM_VOCAB_PATH, MODELS_DIR, PLOTS_DIR, TFIDF_MODEL_PATH, ensure_directories
from shared.dataset import labels, load_companies, split_frames
from shared.logging_utils import get_logger
from shared.metrics import classification_metrics
from shared.nn_models import RiskLSTM
from shared.text_data import build_vocab, save_vocab, texts_to_sequences

logger = get_logger("train_lstm")
MAX_LEN = 160
BATCH_SIZE = 64
EPOCHS = 6


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(sequences: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(sequences)
    tensor_y = torch.from_numpy(y.astype(np.int64))
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def _predict(model: RiskLSTM, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        probs.append(prob)
        preds.append(pred)
    return np.concatenate(preds), np.concatenate(probs)


def main() -> None:
    ensure_directories()
    frame = load_companies()
    train, val, test = split_frames(frame)
    y_train, y_val, y_test = labels(train), labels(val), labels(test)

    vectorizer_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logger.info("Training TF-IDF baseline")
    vectorizer_clf.fit(train["narrative"], y_train)
    tfidf_proba = vectorizer_clf.predict_proba(test["narrative"])[:, 1]
    tfidf_pred = (tfidf_proba >= 0.5).astype(int)
    tfidf_metrics = classification_metrics(y_test, tfidf_pred, tfidf_proba)
    joblib.dump(vectorizer_clf, TFIDF_MODEL_PATH)

    vocab = build_vocab(train["narrative"].tolist())
    save_vocab(vocab, LSTM_VOCAB_PATH)
    train_seq = texts_to_sequences(train["narrative"].tolist(), vocab, MAX_LEN)
    val_seq = texts_to_sequences(val["narrative"].tolist(), vocab, MAX_LEN)
    test_seq = texts_to_sequences(test["narrative"].tolist(), vocab, MAX_LEN)

    device = _device()
    model = RiskLSTM(vocab_size=len(vocab)).to(device)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int((y_train == 0).sum())
    weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = _loader(train_seq, y_train, True)
    val_loader = _loader(val_seq, y_val, False)
    test_loader = _loader(test_seq, y_test, False)

    best_state = None
    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(xb)
        _, val_proba = _predict(model, val_loader, device)
        val_pred = (val_proba >= 0.5).astype(int)
        val_metrics = classification_metrics(y_val, val_pred, val_proba)
        logger.info(
            "epoch %s loss=%.4f val_auc=%.4f val_f1=%.4f",
            epoch,
            running / len(train),
            val_metrics["roc_auc"],
            val_metrics["f1"],
        )
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": best_state, "vocab_size": len(vocab), "max_len": MAX_LEN},
        LSTM_MODEL_PATH,
    )
    pred, proba = _predict(model, test_loader, device)
    lstm_metrics = classification_metrics(y_test, pred, proba)
    payload = {"lstm": lstm_metrics, "tfidf": tfidf_metrics, "vocab_size": len(vocab)}
    (MODELS_DIR / "lstm_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="LSTM")
    RocCurveDisplay.from_predictions(y_test, tfidf_proba, ax=ax, name="TF-IDF")
    ax.set_title("Text models ROC")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "text_evaluation.png", dpi=140)
    plt.close(fig)
    logger.info("LSTM test metrics: %s", lstm_metrics)
    logger.info("TF-IDF test metrics: %s", tfidf_metrics)


if __name__ == "__main__":
    main()
