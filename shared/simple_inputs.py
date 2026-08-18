"""Map a few plain-language inputs onto the 64-ratio model vector.

Unknown ratios are filled with medians from the real UCI dataset so a user
can score a company without uploading documents or knowing Attr codes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import pandas as pd

from shared.dataset import load_companies
from shared.features import FEATURE_COLUMNS
from shared.narratives import _num, generate_narrative


@lru_cache(maxsize=1)
def feature_medians() -> dict[str, float]:
    frame = load_companies()
    medians = frame[FEATURE_COLUMNS].median(numeric_only=True).to_dict()
    return {key: float(value) if pd.notna(value) else 0.0 for key, value in medians.items()}


def company_snapshot(row: Mapping[str, Any]) -> dict[str, str]:
    roa = _num(row.get("Attr1"))
    leverage = _num(row.get("Attr2"))
    current = _num(row.get("Attr4"))
    if roa > 0.03:
        profit = "Profitable"
    elif roa >= 0:
        profit = "Barely profitable"
    else:
        profit = "Losing money"
    if leverage > 0.70:
        debt = "High debt"
    elif leverage > 0.45:
        debt = "Moderate debt"
    else:
        debt = "Low debt"
    if current < 1:
        cash = "Cash shortage"
    elif current < 1.2:
        cash = "Tight cash"
    else:
        cash = "Can pay bills"
    return {
        "profit": profit,
        "debt": debt,
        "cash": cash,
        "blurb": f"{profit} · {debt} · {cash}",
    }


def features_from_simple(simple: Mapping[str, Any]) -> dict[str, float]:
    """Build a 64-length ratio vector from four user-facing numbers."""
    features = feature_medians().copy()
    profit = _num(simple.get("profit"), 0.03)
    debt = _num(simple.get("debt"), 0.50)
    current = _num(simple.get("current_ratio"), 1.20)
    equity = _num(simple.get("equity"), max(0.0, 1.0 - debt))
    working_cap = max(-0.4, min(0.5, current - 1.0)) * 0.25

    features["Attr1"] = profit
    features["Attr2"] = debt
    features["Attr3"] = working_cap
    features["Attr4"] = current
    features["Attr6"] = profit * 2.0
    features["Attr7"] = profit
    features["Attr8"] = equity / max(debt, 0.05)
    features["Attr9"] = 1.1
    features["Attr10"] = equity
    features["Attr18"] = profit
    features["Attr23"] = profit
    features["Attr51"] = min(debt, 0.95)
    features["Attr56"] = max(-0.2, profit)
    return features


def narrative_from_simple(simple: Mapping[str, Any], features: Mapping[str, Any]) -> str:
    return generate_narrative(features)
