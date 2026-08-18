"""Structured-data preprocessing: missing values, outliers, scaling."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from shared.features import FEATURE_COLUMNS


@dataclass
class StructuredPreprocessor:
    imputer: SimpleImputer
    lower: np.ndarray
    upper: np.ndarray
    scaler: RobustScaler
    feature_names: list[str]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy(dtype=np.float64)
        imputed = self.imputer.transform(values)
        clipped = np.clip(imputed, self.lower, self.upper)
        return self.scaler.transform(clipped)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "StructuredPreprocessor":
        return joblib.load(path)


def fit_preprocessor(frame: pd.DataFrame) -> StructuredPreprocessor:
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(values)
    lower = np.nanpercentile(imputed, 1, axis=0)
    upper = np.nanpercentile(imputed, 99, axis=0)
    clipped = np.clip(imputed, lower, upper)
    scaler = RobustScaler()
    scaler.fit(clipped)
    return StructuredPreprocessor(
        imputer=imputer,
        lower=lower,
        upper=upper,
        scaler=scaler,
        feature_names=FEATURE_COLUMNS,
    )
