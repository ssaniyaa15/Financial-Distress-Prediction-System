#!/usr/bin/env python3
"""Download public data and build aligned multimodal artifacts.

Selected dataset
----------------
UCI Polish Companies Bankruptcy, 3rd-year forecast file
(Zieba, Tomczak & Tomczak, 2016; CC BY 4.0).

Why this file
-------------
* Real binary target: bankrupt within 3 years vs still operating.
* 10,503 firm-years and 64 financial ratios (profitability, leverage,
  liquidity, turnover). Labels are not fabricated.
* Direct CSV mirror: Hugging Face `weiqi001/bankruptcy-early-warning-data`.

Modality mapping
----------------
The source file is anonymized: there are no company names, tickers, or
filings, so Polish ratios cannot be joined to SEC 10-Ks or news.

Matching strategy (documented limitation):
1. Structured modality  = the original 64 ratios and official bankruptcy label.
2. Text modality        = MD&A-style narrative generated from that row's ratios.
3. Document modality    = statement PDF whose table cells are those same ratios.
4. Vision modality      = bar-chart image of 10 key ratios from the same row.

Every modality for a company-year therefore shares the same real label.
Sample SEC PDFs are stored separately for OCR smoke tests; they are not
used as training labels because they cannot be joined to this table.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.charts import save_chart
from shared.config import (
    COMPANIES_CSV,
    DOCUMENTS_DIR,
    IMAGES_DIR,
    RAW_CSV,
    RAW_DIR,
    SEC_DIR,
    SPLIT_JSON,
    TEXT_DIR,
    ensure_directories,
)
from shared.features import FEATURE_COLUMNS, TARGET_COLUMN
from shared.logging_utils import get_logger
from shared.narratives import generate_narrative
from shared.pdf_statements import write_statement_pdf

logger = get_logger("prepare")

CSV_URL = (
    "https://huggingface.co/datasets/weiqi001/bankruptcy-early-warning-data/"
    "resolve/main/bankruptcy_year3.csv"
)
DICT_URL = (
    "https://huggingface.co/datasets/weiqi001/bankruptcy-early-warning-data/"
    "resolve/main/data_dictionary.csv"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"

# Small public 10-K HTML snapshots used only to demonstrate OCR on real filings.
SEC_SAMPLES = {
    "sec_sample_notice.txt": (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    ),
}


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "finalFINANCE-academic-project/1.0 (vardaanagrawal)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        destination.write_bytes(response.read())
    logger.info("Downloaded %s -> %s (%s bytes)", url, destination, destination.stat().st_size)


def download_raw() -> Path:
    ensure_directories()
    if not RAW_CSV.exists() or RAW_CSV.stat().st_size < 1000:
        logger.info("Downloading Polish bankruptcy year-3 CSV")
        _download(CSV_URL, RAW_CSV)
    dictionary = RAW_DIR / "data_dictionary.csv"
    if not dictionary.exists():
        try:
            _download(DICT_URL, dictionary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download data dictionary: %s", exc)
    return RAW_CSV


def build_processed_table(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected columns in source CSV, missing: {missing}")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    frame.insert(0, "company_id", [f"PL-{index:05d}" for index in range(len(frame))])
    frame.insert(1, "company_name", frame["company_id"].map(lambda cid: f"Polish Co {cid[3:]}"))
    logger.info("Generating narratives for %s rows", len(frame))
    frame["narrative"] = [generate_narrative(row) for row in frame.to_dict(orient="records")]
    return frame


def make_splits(frame: pd.DataFrame) -> dict[str, list[str]]:
    train_ids, test_ids = train_test_split(
        frame["company_id"],
        test_size=0.20,
        random_state=42,
        stratify=frame[TARGET_COLUMN],
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.15,
        random_state=42,
        stratify=frame.loc[frame["company_id"].isin(train_ids), TARGET_COLUMN],
    )
    splits = {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }
    SPLIT_JSON.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    frame = frame.copy()
    inverse = {cid: split for split, ids in splits.items() for cid in ids}
    frame["split"] = frame["company_id"].map(inverse)
    return splits, frame


def write_sample_artifacts(frame: pd.DataFrame, per_class: int = 40) -> None:
    """Write a stratified sample of PDFs and chart images for OCR / demo."""
    parts = []
    for _, group in frame.groupby(TARGET_COLUMN):
        parts.append(group.sample(n=min(per_class, len(group)), random_state=42))
    samples = pd.concat(parts, ignore_index=True)
    logger.info("Writing %s sample PDFs and charts", len(samples))
    manifest = []
    for row in samples.to_dict(orient="records"):
        company_id = row["company_id"]
        pdf_path = DOCUMENTS_DIR / f"{company_id}.pdf"
        img_path = IMAGES_DIR / f"{company_id}.png"
        write_statement_pdf(row, pdf_path, company_id)
        save_chart(row, img_path)
        text_path = TEXT_DIR / f"{company_id}.txt"
        text_path.write_text(row["narrative"], encoding="utf-8")
        manifest.append(
            {
                "company_id": company_id,
                "bankrupt": int(row[TARGET_COLUMN]),
                "pdf": str(pdf_path.relative_to(ROOT)),
                "image": str(img_path.relative_to(ROOT)),
                "text": str(text_path.relative_to(ROOT)),
            }
        )
    (DOCUMENTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_dataset_card(frame: pd.DataFrame) -> None:
    counts = frame[TARGET_COLUMN].value_counts().to_dict()
    card = {
        "dataset": "UCI Polish Companies Bankruptcy (3rd year)",
        "source_csv": str(RAW_CSV),
        "citation": "Zieba, Tomczak, Tomczak (2016). Expert Systems with Applications.",
        "license": "CC BY 4.0",
        "n_rows": int(len(frame)),
        "n_features": 64,
        "target": "bankrupt (1 = filed within 3 years)",
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "missing_cells": int(frame[FEATURE_COLUMNS].isna().sum().sum()),
        "join_limitation": (
            "Source rows are anonymized Polish firms. Text, PDF, and image "
            "modalities are reconstructed from the same ratio vector so labels stay real."
        ),
        "sec_samples": "Optional OCR smoke-test files in data/sec_samples; not used for training labels.",
    }
    (RAW_DIR / "DATASET_CARD.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    logger.info("Dataset card: %s", card)


def try_fetch_sec_notice() -> None:
    notice = SEC_DIR / "README.txt"
    notice.write_text(
        "SEC EDGAR filings cannot be joined to the anonymized Polish ratio table.\n"
        "Generated statement PDFs in data/documents/ are the training/OCR documents.\n"
        "To OCR a real 10-K, upload it in the frontend; the OCR service accepts any PDF.\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()
    raw_path = download_raw()
    frame = build_processed_table(raw_path)
    _, frame = make_splits(frame)
    COMPANIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPANIES_CSV, index=False)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / "narratives.csv").write_text(
        frame[["company_id", "narrative", TARGET_COLUMN]].to_csv(index=False),
        encoding="utf-8",
    )
    write_sample_artifacts(frame)
    write_dataset_card(frame)
    try_fetch_sec_notice()
    logger.info("Processed table saved to %s (%s rows)", COMPANIES_CSV, len(frame))


if __name__ == "__main__":
    main()
