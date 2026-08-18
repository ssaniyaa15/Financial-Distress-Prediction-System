"""Turn structured ratios into financial-statement style text.

The Polish dataset has no company names or filings. Each narrative is built
from that row's actual ratio values so the text modality stays aligned with
the real bankruptcy label. No labels are invented.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from shared.features import FEATURE_DEFINITIONS, KEY_FEATURE_LABELS, KEY_FEATURES, SHORT_NAMES


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def generate_narrative(row: Mapping[str, Any]) -> str:
    """Create an MD&A-style paragraph from observed financial ratios."""
    roa = _num(row.get("Attr1"))
    leverage = _num(row.get("Attr2"))
    wc = _num(row.get("Attr3"))
    current = _num(row.get("Attr4"))
    retained = _num(row.get("Attr6"))
    ebit = _num(row.get("Attr7"))
    turnover = _num(row.get("Attr9"))
    equity = _num(row.get("Attr10"))
    margin = _num(row.get("Attr23"))
    st_liab = _num(row.get("Attr51"))
    sales_growth = _num(row.get("Attr21"), 1.0)
    coverage = _num(row.get("Attr27"))

    profit_view = (
        "Profitability is weak, with negative or near-zero returns on assets."
        if roa < 0.02
        else "The firm remains profitable on an asset basis."
        if roa < 0.08
        else "Asset returns are solid relative to typical industrial firms."
    )
    leverage_view = (
        "Leverage is elevated: liabilities consume a large share of the balance sheet."
        if leverage > 0.70
        else "Leverage is moderate."
        if leverage > 0.45
        else "The capital structure is conservative, with limited reliance on debt."
    )
    liquidity_view = (
        "Liquidity is tight. Working capital is thin and the current ratio is under pressure."
        if wc < 0.05 or current < 1.0
        else "Short-term liquidity appears adequate."
        if current < 1.5
        else "The company holds a comfortable liquidity buffer."
    )
    equity_view = (
        "Equity coverage of assets is limited, leaving little cushion against losses."
        if equity < 0.25
        else "Shareholders' equity still provides a meaningful buffer."
    )
    growth_view = (
        "Year-over-year sales contracted."
        if sales_growth < 0.95
        else "Sales were roughly stable versus the prior year."
        if sales_growth < 1.08
        else "Sales expanded compared with the prior year."
    )
    coverage_view = (
        "Operating profit provides limited coverage of financial expenses."
        if coverage < 1.5
        else "Interest and financial expenses are covered by operating profit."
    )

    lines = [
        "Management discussion of financial condition.",
        f"Return on assets (net profit / total assets) is {_fmt(roa)}. {profit_view}",
        f"Total liabilities / total assets equals {_fmt(leverage)}. {leverage_view}",
        f"Working capital / total assets is {_fmt(wc)} and the current ratio is {_fmt(current)}. {liquidity_view}",
        f"Retained earnings / total assets stands at {_fmt(retained)}.",
        f"EBIT / total assets is {_fmt(ebit)} and net profit / sales is {_fmt(margin)}.",
        f"Equity / total assets is {_fmt(equity)}. {equity_view}",
        f"Asset turnover (sales / total assets) is {_fmt(turnover)}. {growth_view}",
        f"Short-term liabilities / total assets equals {_fmt(st_liab)}. {coverage_view}",
        (
            "Key statement items in ratio form: "
            f"revenue_proxy=asset_turnover {_fmt(turnover)}, "
            f"profit_proxy={_fmt(roa)}, debt_proxy={_fmt(leverage)}, "
            f"assets_base=1.00, liabilities_proxy={_fmt(leverage)}, "
            f"equity_proxy={_fmt(equity)}."
        ),
    ]
    return " ".join(lines)


def structured_summary(row: Mapping[str, Any]) -> dict[str, float]:
    summary = {}
    for feature in KEY_FEATURES:
        summary[SHORT_NAMES[feature]] = round(_num(row.get(feature)), 6)
    return summary


def contributing_factors(row: Mapping[str, Any], risk_score: float) -> list[str]:
    factors: list[str] = []
    checks = [
        ("Attr2", 0.70, "high", "High leverage (liabilities / assets)"),
        ("Attr1", 0.02, "low", "Weak return on assets"),
        ("Attr3", 0.05, "low", "Thin working capital"),
        ("Attr4", 1.00, "low", "Current ratio below 1"),
        ("Attr10", 0.25, "low", "Low equity buffer"),
        ("Attr23", 0.00, "low", "Negative or zero profit margin"),
        ("Attr51", 0.50, "high", "Heavy short-term liabilities"),
        ("Attr6", 0.00, "low", "Negative retained earnings"),
    ]
    for feature, threshold, direction, label in checks:
        value = _num(row.get(feature))
        triggered = value > threshold if direction == "high" else value < threshold
        if triggered:
            factors.append(f"{label}: {KEY_FEATURE_LABELS[feature]}={_fmt(value)}")
    if not factors:
        if risk_score >= 0.5:
            factors.append("Combined ratio profile is closer to historically distressed firms.")
        else:
            factors.append("Ratio profile is closer to historically healthy firms.")
    return factors[:6]


def extract_financial_fields(text: str) -> dict[str, str]:
    """Pull labeled financial items from OCR or narrative text."""
    import re

    patterns = {
        "revenue": r"(revenue_proxy|sales / total assets|asset turnover)[^\d\-]*(-?\d+\.?\d*)",
        "profit": r"(profit_proxy|return on assets|net profit / total assets)[^\d\-]*(-?\d+\.?\d*)",
        "debt": r"(debt_proxy|leverage|total liabilities / total assets)[^\d\-]*(-?\d+\.?\d*)",
        "assets": r"(assets_base|total assets)[^\d\-]*(-?\d+\.?\d*)",
        "liabilities": r"(liabilities_proxy|total liabilities)[^\d\-]*(-?\d+\.?\d*)",
        "equity": r"(equity_proxy|equity / total assets)[^\d\-]*(-?\d+\.?\d*)",
    }
    found: dict[str, str] = {}
    lowered = text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            found[key] = match.group(2)
    return found
