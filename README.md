# Credit Card Fraud Detection

XGBoost classifier with Isolation Forest anomaly scoring for real-time fraud detection, deployed via Streamlit Community Cloud.

## Project structure

```
.
├── src/
│   ├── features.py          # Feature engineering pipeline (sklearn transformers)
│   ├── anomaly.py           # IsolationForestScorer (appends isolation_score)
│   ├── train.py             # Training script — produces models/*.joblib
│   └── explain.py           # SHAP-based explainability utilities
├── streamlit_app.py         # Streamlit UI — loads models and serves predictions
├── requirements.txt         # Python dependencies for Streamlit Community Cloud
├── data/
│   └── test_data.csv        # Kaggle credit card fraud dataset (not committed)
└── models/                  # Produced by train.py (not committed)
    ├── fraud_pipeline.joblib
    ├── isolation_scorer.joblib
    └── feature_pipeline.joblib
```

## Features

| Group | Columns |
|---|---|
| PCA components | V1 – V28 |
| Amount | Amount, log\_amount, amount\_deviation |
| Time-of-day | hour\_of\_day, day\_of\_week, is\_night, hour\_sin, hour\_cos |
| Velocity | tx\_count\_1h, tx\_count\_24h |
| Anomaly | isolation\_score |

**39 features total.** The isolation score is produced by a fitted `IsolationForestScorer` saved alongside the classifier pipeline.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Place the Kaggle credit card fraud CSV at `data/test_data.csv`, then:

```bash
python -m src.train
```

This runs 5-fold TimeSeriesSplit cross-validation, prints PR-AUC metrics, and writes:

- `models/fraud_pipeline.joblib` — fitted scaler + SMOTE + XGBoost classifier
- `models/isolation_scorer.joblib` — fitted IsolationForestScorer
- `models/feature_pipeline.joblib` — fitted feature engineering pipeline (TimeFeatureTransformer, AmountFeatureTransformer, TransactionVelocityTransformer)

Optional flags:

```
--data PATH          CSV path (default: data/test_data.csv)
--n-splits N         Number of TimeSeriesSplit folds (default: 5)
--sliding-window     Use SlidingWindowSplit instead of TimeSeriesSplit
--output-dir DIR     Directory for PR curve and confusion matrix plots (default: reports)
```

### 3. Run locally

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Enter a transaction manually (Time, V1–V28, Amount) or upload a single-row CSV. The app returns:

- Fraud probability
- Predicted class (0 = Legit, 1 = Fraud)
- Verdict banner
- SHAP waterfall plot showing which features drove the decision

### 4. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (include `streamlit_app.py`, `requirements.txt`, and `src/`).
2. Commit the `models/` directory too, or use [DVC](https://dvc.org/) / [Git LFS](https://git-lfs.com/) to track the `.joblib` files.
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
4. Select your repo, branch, and set **Main file path** to `streamlit_app.py`.
5. In **Advanced settings**, set the Python version to **3.11** or **3.12**.
6. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

> **Note:** `pyarrow` is not listed in `requirements.txt` — Streamlit pulls a compatible version automatically. Do not add an explicit `pyarrow` pin unless a build failure specifically calls it out.

## Model details

- **Algorithm:** XGBoost (`n_estimators=400`, `max_depth=5`, `learning_rate=0.05`)
- **Class imbalance:** SMOTE oversampling of the minority (fraud) class at training time; `scale_pos_weight` set from the neg/pos ratio
- **Validation:** TimeSeriesSplit (respects temporal ordering of transactions)
- **Metric:** PR-AUC (appropriate for highly imbalanced fraud data)
- **Explainability:** SHAP TreeExplainer waterfall plots per transaction
