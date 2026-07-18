from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from src.features import FEATURE_COLS, PCA_COLS

_NEUTRAL = "#6C757D"

ALL_FEATURE_COLS = FEATURE_COLS

class FraudExplainer:
    def __init__(
        self,
        model:         XGBClassifier,
        feature_names: Optional[list[str]] = None,
        output_dir:    str = "reports",
    ) -> None:
        self.model         = model
        self.feature_names = feature_names or ALL_FEATURE_COLS
        self.output_dir    = Path(output_dir)
        self.explainer_:   Optional[shap.TreeExplainer] = None
        self.shap_values_: Optional[np.ndarray]          = None
        self.explanation_: Optional[shap.Explanation]    = None
        self.base_value_:  Optional[float]               = None
    def fit(self, X_test: Union[pd.DataFrame, np.ndarray]) -> "FraudExplainer":
        X_arr, self.feature_names = self._prepare(X_test)
        print("[SHAP] Building TreeExplainer ...")
        self.explainer_ = shap.TreeExplainer(self.model, model_output="raw")
        print(f"[SHAP] Computing SHAP values for {X_arr.shape[0]} samples × {X_arr.shape[1]} features ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_out = self.explainer_(X_arr)
        if isinstance(shap_out.values, np.ndarray) and shap_out.values.ndim == 3:
            values   = shap_out.values[:, :, 1]
            base_val = shap_out.base_values[:, 1] if shap_out.base_values.ndim == 2 else shap_out.base_values
        else:
            values   = shap_out.values
            base_val = shap_out.base_values
        self.shap_values_ = values
        self.base_value_  = float(np.mean(base_val))
        self.explanation_ = shap.Explanation(
            values        = values,
            base_values   = base_val if np.ndim(base_val) > 0 else np.full(len(values), base_val),
            data          = X_arr,
            feature_names = self.feature_names,
        )
        print(f"[SHAP] Done.  Base value (avg model log-odds): {self.base_value_:.4f}")
        return self
    def plot_summary(
        self,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plot_type:   str = "dot",
    ) -> None:
        self._require_fit()
        path = Path(output_path) if output_path else self.output_dir / "shap_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            self.shap_values_,
            self.explanation_.data,
            feature_names   = self.feature_names,
            plot_type       = plot_type,
            max_display     = max_display,
            show            = False,
            plot_size       = None,
            color_bar_label = "Feature value",
        )
        ax = plt.gca()
        ax.set_title(
            "Global Feature Importance — SHAP Summary (Credit Card Fraud)\n"
            "What drives the model's fraud decisions across all transactions?",
            fontsize=12, pad=12,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(0.99, 0.01, "← Reduces fraud score     Increases fraud score →",
                transform=ax.transAxes, fontsize=8, color=_NEUTRAL, ha="right", va="bottom")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Summary plot saved -> {path}")
    def explain_transaction(
        self,
        X_test:     Union[pd.DataFrame, np.ndarray],
        row_index:  int,
        output_dir: Optional[str] = None,
        risk_score: Optional[float] = None,
    ) -> dict:
        self._require_fit()
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        X_arr, _ = self._prepare(X_test)
        if row_index >= len(X_arr):
            raise IndexError(f"row_index {row_index} is out of range for X_test with {len(X_arr)} rows.")
        shap_row    = self.shap_values_[row_index]
        feature_row = X_arr[row_index]
        base_val    = float(self.explanation_.base_values[row_index])
        sorted_idx  = np.argsort(np.abs(shap_row))[::-1]
        top_risk_drivers = [
            (self.feature_names[i], float(shap_row[i]), float(feature_row[i]))
            for i in sorted_idx[:5]
        ]
        single_exp = shap.Explanation(
            values        = shap_row,
            base_values   = base_val,
            data          = feature_row,
            feature_names = self.feature_names,
        )
        risk_label     = f" | Fraud probability: {risk_score:.1%}" if risk_score else ""
        waterfall_path = self._plot_waterfall(single_exp, row_index, risk_label, out_dir)
        force_path     = self._plot_force(shap_row, feature_row, base_val, row_index, risk_label, out_dir)
        self._print_transaction_summary(row_index, risk_score, top_risk_drivers, base_val)
        return {
            "shap_values":      shap_row,
            "feature_values":   dict(zip(self.feature_names, feature_row.tolist())),
            "base_value":       base_val,
            "top_risk_drivers": top_risk_drivers,
            "waterfall_path":   str(waterfall_path),
            "force_path":       str(force_path),
        }
    def _plot_waterfall(self, single_exp, row_index, risk_label, out_dir) -> Path:
        plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(single_exp, max_display=15, show=False)
        ax = plt.gca()
        ax.set_title(
            f"Transaction #{row_index} — SHAP Waterfall{risk_label}\n"
            "How each PCA/engineered feature contributed to the fraud score",
            fontsize=11, pad=10,
        )
        plt.figtext(0.5, -0.02,
                    "Red bars push toward FRAUD | Blue bars push toward SAFE",
                    ha="center", fontsize=9, color=_NEUTRAL)
        path = out_dir / f"waterfall_tx{row_index}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Waterfall plot saved -> {path}")
        return path
    def _plot_force(self, shap_row, feature_row, base_val, row_index, risk_label, out_dir) -> Path:
        shap.plots.force(
            base_value             = base_val,
            shap_values            = shap_row,
            features               = feature_row,
            feature_names          = self.feature_names,
            matplotlib             = True,
            show                   = False,
            figsize                = (18, 4),
            contribution_threshold = 0.02,
        )
        plt.title(
            f"Transaction #{row_index} — Force Plot{risk_label}\n"
            "Red = pushes toward fraud | Blue = pushes toward safe",
            fontsize=10, pad=8,
        )
        plt.tight_layout()
        path = out_dir / f"force_tx{row_index}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Force plot saved -> {path}")
        return path
    def _print_transaction_summary(self, row_index, risk_score, top_risk_drivers, base_val) -> None:
        print(f"\n{'─'*60}")
        print(f"  TRANSACTION #{row_index} — FRAUD EXPLANATION AUDIT TRAIL")
        print(f"{'─'*60}")
        if risk_score is not None:
            verdict = "HIGH RISK" if risk_score >= 0.5 else "MODERATE RISK"
            print(f"  Fraud probability : {risk_score:.2%}  [{verdict}]")
        print(f"  Model base value  : {base_val:.4f}")
        print(f"\n  Top 5 contributing features:")
        print(f"  {'Feature':<28} {'SHAP Value':>12}  {'Raw Value':>12}  Direction")
        print(f"  {'─'*28} {'─'*12}  {'─'*12}  {'─'*9}")
        for feat, shap_val, raw_val in top_risk_drivers:
            direction = "▲ FRAUD" if shap_val > 0 else "▼ SAFE"
            print(f"  {feat:<28} {shap_val:>+12.4f}  {raw_val:>12.4f}  {direction}")
        print(f"{'─'*60}\n")
    def _require_fit(self) -> None:
        if self.shap_values_ is None:
            raise RuntimeError("FraudExplainer has not been fitted. Call .fit(X_test) first.")
    def _prepare(self, X: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32), list(X.columns)
        arr = np.asarray(X, dtype=np.float32)
        names = list(self.feature_names) if self.feature_names \
                else [f"feature_{i}" for i in range(arr.shape[1])]
        return arr, names

def explain_top_flagged(
    explainer:  FraudExplainer,
    X_test:     Union[pd.DataFrame, np.ndarray],
    y_scores:   np.ndarray,
    top_n:      int = 5,
    output_dir: str = "reports/explanations",
) -> list[dict]:
    ranked_idx   = np.argsort(y_scores)[::-1][:top_n]
    explanations = []
    print(f"\n[SHAP] Explaining top {top_n} flagged transactions ...")
    for rank, idx in enumerate(ranked_idx, start=1):
        print(f"\n  [{rank}/{top_n}] Transaction row {idx} — risk score: {y_scores[idx]:.2%}")
        result = explainer.explain_transaction(
            X_test,
            row_index  = int(idx),
            output_dir = output_dir,
            risk_score = float(y_scores[idx]),
        )
        explanations.append(result)
    return explanations

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from src.features import build_feature_pipeline, LABEL_COL, TIME_COL

    print("\n" + "=" * 60)
    print("  FraudExplainer — Credit Card Fraud Dataset Demo")
    print("=" * 60)

    df_raw = pd.read_csv("data/test_data.csv")
    y_all  = df_raw.pop(LABEL_COL).astype(int)
    df_raw = df_raw.sort_values(TIME_COL).reset_index(drop=True)
    y_all  = y_all.reindex(df_raw.index).reset_index(drop=True)

    feat_pipe  = build_feature_pipeline()
    df_eng     = feat_pipe.fit_transform(df_raw)
    feat_cols  = [c for c in FEATURE_COLS if c in df_eng.columns]
    X_all      = df_eng[feat_cols].fillna(0).values.astype(np.float32)
    y_arr      = y_all.values

    split      = int(len(X_all) * 0.75)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

    print(f"\n  Train: {len(X_train)} rows | Fraud: {y_train.sum()}")
    print(f"  Test : {len(X_test)} rows  | Fraud: {y_test.sum()}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model   = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    model.fit(X_tr_sc, y_train, verbose=False)

    y_scores = model.predict_proba(X_te_sc)[:, 1]
    print(f"\n  Model trained.  Max fraud score: {y_scores.max():.4f}")

    X_te_df  = pd.DataFrame(X_te_sc, columns=feat_cols)
    explainer = FraudExplainer(model, feature_names=feat_cols, output_dir="reports")
    explainer.fit(X_te_df)
    explainer.plot_summary("reports/shap_summary.png", max_display=20)

    top_idx = int(np.argmax(y_scores))
    explainer.explain_transaction(X_te_df, row_index=top_idx,
                                  output_dir="reports", risk_score=float(y_scores[top_idx]))
    explain_top_flagged(explainer, X_te_df, y_scores, top_n=3, output_dir="reports/explanations")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60 + "\n")
