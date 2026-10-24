from __future__ import annotations

import numpy as np
import pandas as pd

from shared.features import FEATURE_COLUMNS
from shared.preprocessing import fit_preprocessor


def _frame(n: int, loc: float, scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.normal(loc, scale, n) for col in FEATURE_COLUMNS})


def test_preprocessor_fit_statistics_come_from_train_only() -> None:
    train = _frame(200, loc=0.0, scale=1.0, seed=0)
    test = _frame(80, loc=50.0, scale=1.0, seed=1)
    preprocessor = fit_preprocessor(train)

    train_values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(preprocessor.imputer.statistics_, np.median(train_values, axis=0))
    np.testing.assert_allclose(preprocessor.lower, np.nanpercentile(train_values, 1, axis=0))
    np.testing.assert_allclose(preprocessor.upper, np.nanpercentile(train_values, 99, axis=0))

    stats_before = preprocessor.imputer.statistics_.copy()
    lower_before = preprocessor.lower.copy()
    upper_before = preprocessor.upper.copy()
    center_before = preprocessor.scaler.center_.copy()
    scale_before = preprocessor.scaler.scale_.copy()

    preprocessor.transform(test)

    np.testing.assert_array_equal(preprocessor.imputer.statistics_, stats_before)
    np.testing.assert_array_equal(preprocessor.lower, lower_before)
    np.testing.assert_array_equal(preprocessor.upper, upper_before)
    np.testing.assert_array_equal(preprocessor.scaler.center_, center_before)
    np.testing.assert_array_equal(preprocessor.scaler.scale_, scale_before)


def test_preprocessor_clips_test_outliers_to_train_winsor_bounds() -> None:
    train = _frame(150, loc=0.0, scale=1.0, seed=2)
    test = _frame(10, loc=0.0, scale=1.0, seed=3)
    test.iloc[0, 0] = 1e6
    preprocessor = fit_preprocessor(train)
    transformed = preprocessor.transform(test)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] < 50, "train-fitted winsorizer should clip the 1e6 test outlier"
