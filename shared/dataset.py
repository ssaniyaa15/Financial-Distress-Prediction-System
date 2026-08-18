"""Load processed company records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.config import COMPANIES_CSV, SPLIT_JSON
from shared.features import FEATURE_COLUMNS, TARGET_COLUMN


def load_companies(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or COMPANIES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {csv_path}. Run scripts/download_and_prepare.py first."
        )
    return pd.read_csv(csv_path)


def load_splits() -> dict[str, list[str]]:
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_JSON}")
    return json.loads(SPLIT_JSON.read_text(encoding="utf-8"))


def split_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = load_splits()
    train = frame[frame["company_id"].isin(splits["train"])].copy()
    val = frame[frame["company_id"].isin(splits["val"])].copy()
    test = frame[frame["company_id"].isin(splits["test"])].copy()
    return train, val, test


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS]


def labels(frame: pd.DataFrame):
    return frame[TARGET_COLUMN].astype(int).to_numpy()
