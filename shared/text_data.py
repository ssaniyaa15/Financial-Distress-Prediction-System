"""Text tokenization helpers for the LSTM."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9+\-./]+", re.IGNORECASE)
PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 2, max_size: int = 8000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for token, freq in counts.most_common(max_size - 2):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def texts_to_sequences(
    texts: list[str],
    vocab: dict[str, int],
    max_len: int = 160,
) -> np.ndarray:
    unk = vocab[UNK]
    matrix = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize(text)[:max_len]
        ids = [vocab.get(token, unk) for token in tokens]
        matrix[i, : len(ids)] = ids
    return matrix


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(vocab), encoding="utf-8")


def load_vocab(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))
