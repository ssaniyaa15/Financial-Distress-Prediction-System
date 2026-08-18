"""SQLite / SQLAlchemy store for prediction history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from shared.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[str] = mapped_column(String(64), index=True)
    company_name: Mapped[str] = mapped_column(String(256), default="")
    final_risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def save_prediction(company_id: str, company_name: str, report: dict[str, Any]) -> int:
    engine = get_engine()
    record = PredictionRecord(
        company_id=company_id,
        company_name=company_name,
        final_risk_score=float(report["final_risk_score"]),
        risk_level=str(report["risk_level"]),
        confidence=float(report["confidence"]),
        payload_json=json.dumps(report, default=str),
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return int(record.id)


def list_predictions(limit: int = 25) -> list[dict[str, Any]]:
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "company_id": row.company_id,
                "company_name": row.company_name,
                "final_risk_score": row.final_risk_score,
                "risk_level": row.risk_level,
                "confidence": row.confidence,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
