from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.features import build_feature_pipeline, FEATURE_COLS, LABEL_COL, TIME_COL
from src.anomaly import IsolationForestScorer, ANOMALY_FEATURE_COLS
FINAL_MODEL_PATH           = Path("models/fraud_pipeline.joblib")
FINAL_SCORER_PATH          = Path("models/isolation_scorer.joblib")
FINAL_FEATURE_PIPELINE_PATH = Path("models/feature_pipeline.joblib")
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    y = df.pop(LABEL_COL).astype(int)
    return df, y

class SlidingWindowSplit:
    def __init__(self, train_window: int = 1_500, val_window: int = 300, step: int = 300) -> None:
        self.train_window = train_window
        self.val_window   = val_window
        self.step         = step
    def split(self, X, y=None, groups=None):
        n = len(X)
        start = 0
        while True:
            train_end = start + self.train_window
            val_end   = train_end + self.val_window
            if val_end > n:
                break
            yield np.arange(start, train_end), np.arange(train_end, val_end)
            start += self.step
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n = len(X) if X is not None else 0
        count, start = 0, 0
        while True:
            if start + self.train_window + self.val_window > n:
                break
            count += 1
            start += self.step
        return count

def build_model_pipeline(scale_pos_weight: float = 1.0) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=RANDOM_STATE)),
            ("classifier", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

def cross_validate_model(X: np.ndarray, y: np.ndarray, cv, *, verbose: bool = True) -> dict:
    fold_pr_aucs, all_y_true, all_y_scores, all_y_pred = [], [], [], []
    splits  = list(cv.split(X, y))
    n_folds = len(splits)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Time-Based Cross-Validation  ({n_folds} folds)")
        print(f"{'='*60}")
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        pipeline = build_model_pipeline(scale_pos_weight=neg / max(pos, 1))
        pipeline.fit(X_train, y_train)
        y_score = pipeline.predict_proba(X_val)[:, 1]
        y_pred  = pipeline.predict(X_val)
        pr_auc  = average_precision_score(y_val, y_score)
        fold_pr_aucs.append(pr_auc)
        all_y_true.extend(y_val.tolist())
        all_y_scores.extend(y_score.tolist())
        all_y_pred.extend(y_pred.tolist())
        if verbose:
            print(f"  Fold {fold_idx:>2}/{n_folds} | Train: {len(train_idx):>6} | "
                  f"Val: {len(val_idx):>5} | Fraud in val: {int(y_val.sum()):>3} | PR-AUC: {pr_auc:.4f}")
    if verbose:
        print(f"{'─'*60}")
        print(f"  Mean PR-AUC : {np.mean(fold_pr_aucs):.4f}  (+/- {np.std(fold_pr_aucs):.4f})")
        print(f"{'='*60}\n")
    return {
        "fold_pr_aucs": fold_pr_aucs,
        "all_y_true":   np.array(all_y_true),
        "all_y_scores": np.array(all_y_scores),
        "all_y_pred":   np.array(all_y_pred),
    }

def plot_precision_recall_curve(y_true, y_scores, output_path="reports/pr_curve.png") -> float:
    pr_auc = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recall, precision, color="#6C63FF", lw=2, where="post", label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, step="post", alpha=0.15, color="#6C63FF")
    prevalence = y_true.mean()
    ax.axhline(y=prevalence, color="#FF6584", lw=1.5, linestyle="--",
               label=f"Random baseline (prevalence = {prevalence:.4f})")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve\n(pooled across all CV folds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] PR curve  -> {output_path}")
    return pr_auc

def plot_confusion_matrix(y_true, y_pred, output_path="reports/confusion_matrix.png", threshold=0.5) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold})\npooled across all CV folds", fontsize=13)
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight("bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Confusion matrix -> {output_path}")

def print_summary(fold_pr_aucs, y_true, y_pred, overall_pr_auc) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    print("\n" + "=" * 50)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  Folds evaluated    : {len(fold_pr_aucs)}")
    print(f"  Mean PR-AUC        : {np.mean(fold_pr_aucs):.4f}")
    print(f"  Std  PR-AUC        : {np.std(fold_pr_aucs):.4f}")
    print(f"  Min  PR-AUC        : {np.min(fold_pr_aucs):.4f}")
    print(f"  Max  PR-AUC        : {np.max(fold_pr_aucs):.4f}")
    print(f"\n  --- Pooled Validation Metrics ---")
    print(f"  Overall PR-AUC     : {overall_pr_auc:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"\n  Confusion Matrix (pooled):")
    print(f"    TN={tn:<6} FP={fp}")
    print(f"    FN={fn:<6} TP={tp}")
    print("=" * 50 + "\n")

def main(
    csv_path:           str  = "data/test_data.csv",
    n_splits:           int  = 5,
    use_sliding_window: bool = False,
    output_dir:         str  = "reports",
) -> None:
    print(f"\n[INFO] Loading dataset: {csv_path}")
    raw_df, y = load_data(csv_path)
    print(f"[INFO] Dataset shape : {raw_df.shape}")
    print(f"[INFO] Fraud rate    : {y.mean()*100:.4f}%  ({y.sum()} fraud / {len(y)} total)")

    raw_df = raw_df.sort_values(TIME_COL).reset_index(drop=True)
    y      = y.reindex(raw_df.index).reset_index(drop=True)

    print("\n[INFO] Running feature engineering pipeline ...")
    feat_pipeline = build_feature_pipeline()
    df_features   = feat_pipeline.fit_transform(raw_df)

    print("[INFO] Fitting IsolationForestScorer ...")
    anomaly_cols = [c for c in ANOMALY_FEATURE_COLS if c in df_features.columns]
    scorer = IsolationForestScorer()
    scorer.fit(df_features[anomaly_cols], y.values)
    df_scored = scorer.transform(df_features[anomaly_cols])
    for col in df_features.columns:
        if col not in df_scored.columns:
            df_scored[col] = df_features[col].values

    present_cols = [c for c in FEATURE_COLS if c in df_scored.columns]
    X = df_scored[present_cols].fillna(0).values.astype(np.float32)
    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Features used: {present_cols}")

    if use_sliding_window:
        print("\n[INFO] CV strategy: SlidingWindowSplit (fixed train window)")
        train_win = max(int(len(X) * 0.45), 1000)
        val_win   = max(int(len(X) * 0.10), 500)
        cv = SlidingWindowSplit(train_window=train_win, val_window=val_win, step=val_win)
        print(f"         train_window={train_win} | val_window={val_win}")
    else:
        print(f"\n[INFO] CV strategy: TimeSeriesSplit (n_splits={n_splits})")
        cv = TimeSeriesSplit(n_splits=n_splits)

    results = cross_validate_model(X, y.values, cv, verbose=True)

    overall_pr_auc = plot_precision_recall_curve(
        results["all_y_true"], results["all_y_scores"],
        output_path=f"{output_dir}/pr_curve.png"
    )
    plot_confusion_matrix(
        results["all_y_true"], results["all_y_pred"],
        output_path=f"{output_dir}/confusion_matrix.png"
    )
    print_summary(results["fold_pr_aucs"], results["all_y_true"], results["all_y_pred"], overall_pr_auc)
    print("\n[INFO] Fitting final pipeline on full dataset ...")
    y_arr = y.values
    neg_all = (y_arr == 0).sum()
    pos_all = (y_arr == 1).sum()
    final_pipeline = build_model_pipeline(scale_pos_weight=neg_all / max(pos_all, 1))
    final_pipeline.fit(X, y_arr)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, FINAL_MODEL_PATH, compress=3)
    FINAL_SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, FINAL_SCORER_PATH, compress=3)
    FINAL_FEATURE_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feat_pipeline, FINAL_FEATURE_PIPELINE_PATH, compress=3)
    print(f"[INFO] Model saved            → {FINAL_MODEL_PATH.resolve()}")
    print(f"[INFO] Scorer saved           → {FINAL_SCORER_PATH.resolve()}")
    print(f"[INFO] Feature pipeline saved → {FINAL_FEATURE_PIPELINE_PATH.resolve()}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector on Kaggle CC fraud dataset.")
    parser.add_argument("--data",           type=str,  default="data/test_data.csv")
    parser.add_argument("--n-splits",       type=int,  default=5)
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--output-dir",     type=str,  default="reports")
    args = parser.parse_args()
    main(
        csv_path=args.data,
        n_splits=args.n_splits,
        use_sliding_window=args.sliding_window,
        output_dir=args.output_dir,
    )
