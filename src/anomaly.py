from __future__ import annotations
import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted
from src.features import FEATURE_COLS, PCA_COLS, AMOUNT_COL, TIME_COL, LABEL_COL

ANOMALY_FEATURE_COLS = PCA_COLS + ["Amount", "log_amount", "amount_deviation",
                                    "hour_of_day", "hour_sin", "hour_cos",
                                    "tx_count_1h", "tx_count_24h"]

class IsolationForestScorer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_estimators:  int                   = 200,
        max_samples:   Union[str, int, float] = "auto",
        contamination: Union[str, float]      = 0.002,
        max_features:  float                  = 1.0,
        bootstrap:     bool                   = False,
        score_col:     str                    = "isolation_score",
        random_state:  int                    = 42,
    ) -> None:
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.contamination = contamination
        self.max_features  = max_features
        self.bootstrap     = bootstrap
        self.score_col     = score_col
        self.random_state  = random_state
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
    ) -> "IsolationForestScorer":
        X_arr, feature_names = self._to_array(X)
        if y is not None:
            y_arr = np.asarray(y).ravel()
            if len(y_arr) != len(X_arr):
                raise ValueError(f"X and y length mismatch: {len(X_arr)} vs {len(y_arr)}")
            normal_mask = y_arr == 0
            n_normal    = normal_mask.sum()
            if n_normal == 0:
                raise ValueError("No normal (label=0) samples found in y.")
            X_normal = X_arr[normal_mask]
            if n_normal < 10:
                warnings.warn(f"Only {n_normal} normal samples found.", UserWarning, stacklevel=2)
        else:
            warnings.warn("y is None — training on ALL rows including fraud.", UserWarning, stacklevel=2)
            X_normal = X_arr
            n_normal = len(X_arr)
        self.model_ = IsolationForest(
            n_estimators  = self.n_estimators,
            max_samples   = self.max_samples,
            contamination = self.contamination,
            max_features  = self.max_features,
            bootstrap     = self.bootstrap,
            random_state  = self.random_state,
            n_jobs        = -1,
        )
        self.model_.fit(X_normal)
        self.n_features_in_    = X_arr.shape[1]
        self.feature_names_in_ = feature_names
        self.n_normal_samples_ = int(n_normal)
        return self
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
    ) -> pd.DataFrame:
        check_is_fitted(self, attributes=["model_", "n_features_in_"])
        X_arr, _ = self._to_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")
        anomaly_score = -self.model_.score_samples(X_arr)
        if isinstance(X, pd.DataFrame):
            df_out = X.copy()
        else:
            cols = self.feature_names_in_ if self.feature_names_in_ is not None \
                   else [f"feature_{i}" for i in range(X_arr.shape[1])]
            df_out = pd.DataFrame(X_arr, columns=cols)
        df_out[self.score_col] = anomaly_score
        return df_out
    def score_only(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, attributes=["model_"])
        X_arr, _ = self._to_array(X)
        return -self.model_.score_samples(X_arr)
    def predict_anomaly(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, attributes=["model_"])
        X_arr, _ = self._to_array(X)
        raw = self.model_.predict(X_arr)
        return np.where(raw == -1, 1, 0)
    @staticmethod
    def _to_array(X: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, Optional[list[str]]]:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32), list(X.columns)
        return np.asarray(X, dtype=np.float32), None

def append_anomaly_scores(
    df_train:      pd.DataFrame,
    df_test:       pd.DataFrame,
    y_train:       Union[pd.Series, np.ndarray],
    feature_cols:  list[str] = None,
    scorer_kwargs: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, IsolationForestScorer]:
    if feature_cols is None:
        feature_cols = [c for c in ANOMALY_FEATURE_COLS if c in df_train.columns]
    kwargs = scorer_kwargs or {}
    scorer = IsolationForestScorer(**kwargs)
    scorer.fit(df_train[feature_cols], y_train)
    df_train_out = scorer.transform(df_train[feature_cols])
    df_test_out  = scorer.transform(df_test[feature_cols])
    extra_train = [c for c in df_train.columns if c not in feature_cols]
    extra_test  = [c for c in df_test.columns  if c not in feature_cols]
    if extra_train:
        df_train_out = pd.concat(
            [df_train[extra_train].reset_index(drop=True), df_train_out.reset_index(drop=True)], axis=1
        )
    if extra_test:
        df_test_out = pd.concat(
            [df_test[extra_test].reset_index(drop=True), df_test_out.reset_index(drop=True)], axis=1
        )
    print(f"[IsolationForest] Trained on {scorer.n_normal_samples_} normal samples.")
    print(f"[IsolationForest] Score column '{scorer.score_col}' appended to train and test.")
    print(f"  Train score stats: min={df_train_out[scorer.score_col].min():.4f}  "
          f"median={df_train_out[scorer.score_col].median():.4f}  "
          f"max={df_train_out[scorer.score_col].max():.4f}")
    print(f"  Test  score stats: min={df_test_out[scorer.score_col].min():.4f}  "
          f"median={df_test_out[scorer.score_col].median():.4f}  "
          f"max={df_test_out[scorer.score_col].max():.4f}")
    return df_train_out, df_test_out, scorer

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.features import build_feature_pipeline

    print("\n" + "=" * 60)
    print("  IsolationForestScorer — Credit Card Fraud Demo")
    print("=" * 60)

    df_raw = pd.read_csv("data/test_data.csv")
    y_all  = df_raw.pop(LABEL_COL).astype(int)
    df_raw = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    y_all  = y_all.reindex(df_raw.index).reset_index(drop=True)

    feat_pipe = build_feature_pipeline()
    df_eng    = feat_pipe.fit_transform(df_raw)

    split_idx = int(len(df_eng) * 0.75)
    df_train  = df_eng.iloc[:split_idx].reset_index(drop=True)
    df_test   = df_eng.iloc[split_idx:].reset_index(drop=True)
    y_train   = y_all.values[:split_idx]
    y_test    = y_all.values[split_idx:]

    print(f"\n  Train: {len(df_train)} rows | Fraud: {y_train.sum()}")
    print(f"  Test : {len(df_test)} rows | Fraud: {y_test.sum()}")

    df_train_scored, df_test_scored, scorer = append_anomaly_scores(
        df_train, df_test, y_train,
        scorer_kwargs={"n_estimators": 100, "contamination": 0.002},
    )

    scores_test        = df_test_scored["isolation_score"].values
    test_scores_normal = scores_test[y_test == 0]
    test_scores_fraud  = scores_test[y_test == 1]
    print(f"\n  Score comparison — Normal mean: {test_scores_normal.mean():.4f} | "
          f"Fraud mean: {test_scores_fraud.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(test_scores_normal, bins=60, alpha=0.6, color="#4ECDC4", label="Normal", density=True)
    axes[0].hist(test_scores_fraud,  bins=60, alpha=0.7, color="#FF6B6B", label="Fraud",  density=True)
    axes[0].set_xlabel("Isolation Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Anomaly Score Distribution (Test Set)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].scatter(df_test["Amount"].values[y_test == 0], test_scores_normal,
                    alpha=0.15, s=8, c="#4ECDC4", label="Normal")
    axes[1].scatter(df_test["Amount"].values[y_test == 1], test_scores_fraud,
                    alpha=0.8, s=30, c="#FF6B6B", label="Fraud", zorder=5)
    axes[1].set_xlabel("Transaction Amount")
    axes[1].set_ylabel("Isolation Score")
    axes[1].set_title("Amount vs Anomaly Score (Test Set)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = Path("reports/anomaly_scores.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] Distribution plot -> {out_path}")
    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60 + "\n")
