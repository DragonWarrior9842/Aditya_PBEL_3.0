from __future__ import annotations
import warnings
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

PCA_COLS = [f"V{i}" for i in range(1, 29)]
AMOUNT_COL = "Amount"
TIME_COL = "Time"
LABEL_COL = "Class"

def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_col: str = TIME_COL) -> None:
        self.time_col = time_col
    def fit(self, X: pd.DataFrame, y=None) -> "TimeFeatureTransformer":
        _validate_columns(X, [self.time_col])
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        _validate_columns(X, [self.time_col])
        df = X.copy()
        seconds = df[self.time_col]
        df["hour_of_day"] = (seconds // 3600 % 24).astype(np.int8)
        df["day_of_week"] = (seconds // 86400 % 7).astype(np.int8)
        df["is_night"] = df["hour_of_day"].apply(lambda h: int(h >= 22 or h <= 5))
        hour_rad = 2 * np.pi * df["hour_of_day"] / 24
        df["hour_sin"] = np.sin(hour_rad).round(6)
        df["hour_cos"] = np.cos(hour_rad).round(6)
        return df

class AmountFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, amount_col: str = AMOUNT_COL) -> None:
        self.amount_col = amount_col
        self.global_median_: Optional[float] = None
    def fit(self, X: pd.DataFrame, y=None) -> "AmountFeatureTransformer":
        _validate_columns(X, [self.amount_col])
        self.global_median_ = float(X[self.amount_col].median())
        return self
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.global_median_ is None:
            raise RuntimeError("AmountFeatureTransformer has not been fitted yet.")
        _validate_columns(X, [self.amount_col])
        df = X.copy()
        df["log_amount"] = np.log1p(df[self.amount_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["amount_deviation"] = np.where(
                self.global_median_ == 0,
                1.0,
                df[self.amount_col] / self.global_median_,
            )
        return df

class TransactionVelocityTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_col: str = TIME_COL,
        window_1h: int = 3600,
        window_24h: int = 86400,
    ) -> None:
        self.time_col = time_col
        self.window_1h = window_1h
        self.window_24h = window_24h
    def fit(self, X: pd.DataFrame, y=None) -> "TransactionVelocityTransformer":
        _validate_columns(X, [self.time_col])
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        _validate_columns(X, [self.time_col])
        df = X.copy()
        original_index = df.index
        df = df.sort_values(self.time_col).reset_index(drop=False)
        times = df[self.time_col].values.astype(np.float64)
        df["tx_count_1h"]  = self._rolling_count(times, self.window_1h)
        df["tx_count_24h"] = self._rolling_count(times, self.window_24h)
        df = df.set_index("index").reindex(original_index)
        return df
    @staticmethod
    def _rolling_count(times: np.ndarray, window: float) -> np.ndarray:
        counts = np.zeros(len(times), dtype=np.int64)
        for i in range(len(times)):
            cutoff = times[i] - window
            counts[i] = int(np.sum((times[:i] > cutoff) & (times[:i] < times[i])))
        return counts

def build_feature_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("time_features",   TimeFeatureTransformer()),
            ("amount_features", AmountFeatureTransformer()),
            ("velocity",        TransactionVelocityTransformer()),
        ]
    )

FEATURE_COLS = PCA_COLS + [
    "Amount",
    "log_amount",
    "amount_deviation",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "hour_sin",
    "hour_cos",
    "tx_count_1h",
    "tx_count_24h",
    "isolation_score",
]

if __name__ == "__main__":
    import textwrap
    df = pd.read_csv("data/test_data.csv")
    print(f"Loaded dataset: {df.shape}")
    pipeline = build_feature_pipeline()
    df_out = pipeline.fit_transform(df)
    new_cols = [c for c in df_out.columns if c not in df.columns]
    print(textwrap.dedent(f"""
    Feature Engineering -- Smoke Test Results
    ------------------------------------------
    Input  shape : {df.shape}
    Output shape : {df_out.shape}
    New features : {new_cols}
    """))
    print(df_out[["Time", "Amount"] + new_cols].head(5).to_string(index=False))
    print("\nAll transformers executed successfully.")
