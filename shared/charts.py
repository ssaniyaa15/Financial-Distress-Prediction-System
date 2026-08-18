"""Render financial-ratio charts as images for the vision modality.

Charts are drawn with NumPy so training does not depend on a display backend.
Bar heights encode the actual ratio values; color is fixed per metric so the
label cannot leak through styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from shared.features import KEY_FEATURES, SHORT_NAMES
from shared.narratives import _num

CHART_SIZE = 128
BAR_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def _normalize_for_bar(feature: str, value: float) -> float:
    """Map a raw ratio onto [0, 1] for bar height without using the label."""
    bounds = {
        "Attr1": (-0.3, 0.3),
        "Attr2": (0.0, 1.2),
        "Attr3": (-0.4, 0.6),
        "Attr4": (0.0, 4.0),
        "Attr6": (-0.5, 0.6),
        "Attr7": (-0.3, 0.3),
        "Attr9": (0.0, 3.0),
        "Attr10": (-0.2, 1.0),
        "Attr23": (-0.4, 0.4),
        "Attr51": (0.0, 1.0),
    }
    low, high = bounds[feature]
    if high == low:
        return 0.5
    scaled = (value - low) / (high - low)
    return float(np.clip(scaled, 0.0, 1.0))


def render_ratio_chart(row: Mapping[str, float], size: int = CHART_SIZE) -> np.ndarray:
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    image[:18, :] = (236, 240, 245)
    image[-12:, :] = (220, 224, 230)
    n_bars = len(KEY_FEATURES)
    margin = 10
    usable = size - 2 * margin
    bar_space = usable / n_bars
    bar_width = max(int(bar_space * 0.7), 4)
    axis_y = size - 14

    for index, feature in enumerate(KEY_FEATURES):
        value = _num(row.get(feature))
        height_ratio = _normalize_for_bar(feature, value)
        bar_h = max(int(height_ratio * (size - 40)), 1)
        x0 = margin + int(index * bar_space) + 1
        x1 = min(x0 + bar_width, size - 4)
        y0 = axis_y - bar_h
        color = BAR_COLORS[index % len(BAR_COLORS)]
        image[y0:axis_y, x0:x1] = color
        # small identity tick so the CNN can distinguish columns
        tick = 4 + (index % 3)
        image[axis_y : axis_y + 3, x0 : x0 + tick] = (40, 40, 40)

    # right-side heatmap strip of the same 10 values
    strip_x = size - 8
    cell_h = max((size - 24) // n_bars, 2)
    for index, feature in enumerate(KEY_FEATURES):
        intensity = int(_normalize_for_bar(feature, _num(row.get(feature))) * 180) + 40
        y0 = 18 + index * cell_h
        y1 = min(y0 + cell_h - 1, size - 14)
        image[y0:y1, strip_x - 5 : strip_x] = (intensity, 80, 255 - intensity)
    return image


def chart_to_chw(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def save_chart(row: Mapping[str, float], path: Path) -> Path:
    image = render_ratio_chart(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def chart_feature_vector(row: Mapping[str, float]) -> dict[str, float]:
    return {SHORT_NAMES[feature]: _num(row.get(feature)) for feature in KEY_FEATURES}
