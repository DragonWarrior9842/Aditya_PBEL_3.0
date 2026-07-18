from __future__ import annotations
import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from src.features import FEATURE_COLS, PCA_COLS, TIME_COL, AMOUNT_COL
from src.anomaly import ANOMALY_FEATURE_COLS

warnings.filterwarnings("ignore")

MODEL_PATH            = "models/fraud_pipeline.joblib"
SCORER_PATH           = "models/isolation_scorer.joblib"
FEATURE_PIPELINE_PATH = "models/feature_pipeline.joblib"

@st.cache_resource
def load_artifacts():
    pipeline     = joblib.load(MODEL_PATH)
    scorer       = joblib.load(SCORER_PATH)
    feat_pipeline = joblib.load(FEATURE_PIPELINE_PATH)
    return pipeline, scorer, feat_pipeline

def run_inference(raw_row: pd.DataFrame, pipeline, scorer, feat_pipeline):
    df_eng       = feat_pipeline.transform(raw_row)
    anomaly_cols = [c for c in ANOMALY_FEATURE_COLS if c in df_eng.columns]
    df_scored    = scorer.transform(df_eng[anomaly_cols])
    for col in df_eng.columns:
        if col not in df_scored.columns:
            df_scored[col] = df_eng[col].values
    present = [c for c in FEATURE_COLS if c in df_scored.columns]
    X = df_scored[present].fillna(0).values.astype(np.float32)
    prob = float(pipeline.predict_proba(X)[0, 1])
    pred = int(pipeline.predict(X)[0])
    return prob, pred, X, present

def shap_waterfall(pipeline, X_row: np.ndarray, feature_names: list[str]) -> plt.Figure:
    classifier = pipeline.named_steps["classifier"]
    scaler     = pipeline.named_steps["scaler"]
    X_scaled   = scaler.transform(X_row)
    explainer  = shap.TreeExplainer(classifier, model_output="raw")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_out = explainer(X_scaled)
    if shap_out.values.ndim == 3:
        values   = shap_out.values[0, :, 1]
        base_val = float(shap_out.base_values[0, 1])
    else:
        values   = shap_out.values[0]
        base_val = float(shap_out.base_values[0])
    exp = shap.Explanation(
        values        = values,
        base_values   = base_val,
        data          = X_scaled[0],
        feature_names = feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()
    return fig

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Credit Card Fraud Detection")
st.caption("Powered by XGBoost + Isolation Forest anomaly scoring")

try:
    pipeline, scorer, feat_pipeline = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model artifacts not found: {e}\n\nRun `python -m src.train` first to produce `models/fraud_pipeline.joblib`, `models/isolation_scorer.joblib`, and `models/feature_pipeline.joblib`.")
    st.stop()

tab_form, tab_csv = st.tabs(["Manual Input", "Upload CSV Row"])

with tab_form:
    st.subheader("Enter transaction fields")
    with st.form("transaction_form"):
        col_time, col_amt = st.columns(2)
        with col_time:
            time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
        with col_amt:
            amt_val = st.number_input("Amount (USD)", value=0.0, min_value=0.0, step=0.01, format="%.2f")
        st.markdown("**PCA Components V1–V28** (leave at 0.0 for a typical-looking transaction)")
        v_cols = st.columns(7)
        v_vals = []
        for i, col in enumerate(PCA_COLS):
            idx = i % 7
            with v_cols[idx]:
                v_vals.append(st.number_input(col, value=0.0, step=0.01, format="%.4f", key=f"v_{col}"))
        submitted = st.form_submit_button("Analyse transaction", use_container_width=True)
    if submitted:
        row = {TIME_COL: [time_val], AMOUNT_COL: [amt_val]}
        for col, val in zip(PCA_COLS, v_vals):
            row[col] = [val]
        raw_row = pd.DataFrame(row)
        with st.spinner("Running inference ..."):
            prob, pred, X_row, feat_names = run_inference(raw_row, pipeline, scorer, feat_pipeline)
        verdict = "🔴 HIGH RISK — FRAUD FLAGGED" if pred == 1 else "🟢 LOW RISK — LEGITIMATE"
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud probability", f"{prob:.2%}")
        r2.metric("Predicted class", f"{pred}  ({'Fraud' if pred == 1 else 'Legit'})")
        r3.metric("Verdict", "HIGH RISK" if pred == 1 else "LOW RISK")
        if pred == 1:
            st.error(verdict)
        else:
            st.success(verdict)
        with st.expander("SHAP feature explanation", expanded=True):
            fig = shap_waterfall(pipeline, X_row, feat_names)
            st.pyplot(fig)
            plt.close(fig)

with tab_csv:
    st.subheader("Upload a single-row CSV")
    st.caption("CSV must contain columns: Time, V1–V28, Amount (Class column is ignored if present).")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
            df_upload.columns = [c.strip() for c in df_upload.columns]
            if "Class" in df_upload.columns:
                df_upload = df_upload.drop(columns=["Class"])
            df_upload = df_upload.head(1)
            st.dataframe(df_upload)
            with st.spinner("Running inference ..."):
                prob, pred, X_row, feat_names = run_inference(df_upload, pipeline, scorer, feat_pipeline)
            verdict = "🔴 HIGH RISK — FRAUD FLAGGED" if pred == 1 else "🟢 LOW RISK — LEGITIMATE"
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Fraud probability", f"{prob:.2%}")
            r2.metric("Predicted class", f"{pred}  ({'Fraud' if pred == 1 else 'Legit'})")
            r3.metric("Verdict", "HIGH RISK" if pred == 1 else "LOW RISK")
            if pred == 1:
                st.error(verdict)
            else:
                st.success(verdict)
            with st.expander("SHAP feature explanation", expanded=True):
                fig = shap_waterfall(pipeline, X_row, feat_names)
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            st.error(f"Failed to process uploaded file: {exc}")
