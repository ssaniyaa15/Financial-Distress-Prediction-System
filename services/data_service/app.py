from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.dataset import load_companies
from shared.features import FEATURE_COLUMNS, FEATURE_DEFINITIONS, KEY_FEATURE_LABELS
from shared.json_utils import to_jsonable
from shared.logging_utils import get_logger
from shared.store import list_predictions

logger = get_logger("data_service")
app = FastAPI(title="Financial Risk Data Service", version="1.0.0")


class CompanyQuery(BaseModel):
    limit: int = 50
    split: str | None = None
    bankrupt: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "data"}


@app.get("/features")
def features() -> dict:
    return {"columns": FEATURE_COLUMNS, "definitions": FEATURE_DEFINITIONS, "key_labels": KEY_FEATURE_LABELS}


@app.get("/companies")
def companies(limit: int = 50, split: str | None = None, bankrupt: int | None = None) -> dict:
    frame = load_companies()
    if split:
        frame = frame[frame["split"] == split]
    if bankrupt is not None:
        frame = frame[frame["bankrupt"] == bankrupt]
    subset = frame.head(limit)[
        ["company_id", "company_name", "bankrupt", "split", "Attr1", "Attr2", "Attr3", "Attr4", "Attr10"]
    ]
    return to_jsonable({"count": int(len(frame)), "items": subset.to_dict(orient="records")})


@app.get("/companies/{company_id}")
def company(company_id: str) -> dict:
    frame = load_companies()
    hit = frame[frame["company_id"] == company_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail="Company not found")
    row = hit.iloc[0].to_dict()
    return to_jsonable(row)


@app.get("/history")
def history(limit: int = 25) -> dict:
    return {"items": list_predictions(limit)}
