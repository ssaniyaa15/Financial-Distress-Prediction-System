"""Generate simple financial-statement PDFs from observed ratios."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from shared.features import FEATURE_DEFINITIONS, KEY_FEATURES, KEY_FEATURE_LABELS
from shared.narratives import _fmt, _num, generate_narrative


def write_statement_pdf(row: Mapping[str, object], path: Path, company_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Financial Statement Extract — {company_id}", styles["Title"]),
        Paragraph(
            "Derived from UCI Polish Companies Bankruptcy ratios. "
            "Items are ratio-form statement fields, not reconstructed currency amounts.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    table_data = [["Item", "Ratio value", "Definition"]]
    for feature in KEY_FEATURES:
        table_data.append(
            [
                KEY_FEATURE_LABELS[feature],
                _fmt(_num(row.get(feature))),
                FEATURE_DEFINITIONS[feature],
            ]
        )
    table_data.extend(
        [
            ["Revenue proxy (sales / assets)", _fmt(_num(row.get("Attr9"))), FEATURE_DEFINITIONS["Attr9"]],
            ["Profit proxy (net profit / assets)", _fmt(_num(row.get("Attr1"))), FEATURE_DEFINITIONS["Attr1"]],
            ["Debt proxy (liabilities / assets)", _fmt(_num(row.get("Attr2"))), FEATURE_DEFINITIONS["Attr2"]],
            ["Assets base", "1.0000", "Ratios are scaled by total assets"],
            ["Liabilities proxy", _fmt(_num(row.get("Attr2"))), FEATURE_DEFINITIONS["Attr2"]],
            ["Equity proxy", _fmt(_num(row.get("Attr10"))), FEATURE_DEFINITIONS["Attr10"]],
        ]
    )
    table = Table(table_data, colWidths=[180, 80, 250])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Risk-related discussion", styles["Heading2"]))
    story.append(Paragraph(generate_narrative(row), styles["Normal"]))
    document.build(story)
    return path
