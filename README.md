# 🔍 Credit Card Fraud Detection

A real-time fraud scoring engine that combines **supervised learning**, **unsupervised anomaly detection**, and **explainable AI** into a single Streamlit app — built to catch the 0.17% needle in a 284,807-transaction haystack.

---

## ✨ What it does

Given a single transaction, the app returns a fraud probability, a verdict, and a visual breakdown of *why* the model made that call — in under a second, no cloud dependency required.

- 🎯 **Fraud probability** — a calibrated 0–100% score, not just a binary flag
- 🚦 **Verdict banner** — instant HIGH RISK / LOW RISK read
- 🌊 **SHAP waterfall plot** — see exactly which features pushed the decision, and by how much
- ✍️ **Manual entry or CSV upload** — test a hand-crafted transaction or drop in a real row from the dataset

---

## 🧠 The model

- 🌲 **XGBoost classifier** — gradient-boosted trees (`n_estimators=400`, `max_depth=5`), tuned for a needle-in-a-haystack problem where fraud is 0.17% of all transactions
- ⚖️ **SMOTE oversampling** — synthetically balances the minority (fraud) class during training so the model doesn't just learn to always guess "legit"
- 🕵️ **Isolation Forest anomaly scoring** — a second, unsupervised model trained only on *normal* transactions, contributing an `isolation_score` feature that flags "this doesn't look like anything I've seen before" even for fraud patterns the classifier hasn't explicitly learned
- 📐 **StandardScaler** — normalizes all 39 features before they hit the classifier
- ⏱️ **TimeSeriesSplit cross-validation** — respects the chronological order of transactions instead of shuffling randomly, so the model is never accidentally trained on the future to predict the past
- 📊 **PR-AUC as the north star metric** — precision-recall, not accuracy, because accuracy is meaningless when 99.83% of transactions are legitimate anyway

---

## 🛠️ Feature engineering

39 engineered features feed the model, built entirely with custom `scikit-learn` transformers chained into a single reusable `Pipeline`:

| Category | Features | What it captures |
|---|---|---|
| 🧬 PCA components | `V1`–`V28` | Anonymized transaction attributes from the original dataset |
| 💰 Amount signals | `Amount`, `log_amount`, `amount_deviation` | Raw spend, log-scaled spend, and deviation from the typical transaction size |
| 🌙 Time-of-day | `hour_of_day`, `day_of_week`, `is_night`, `hour_sin`, `hour_cos` | Cyclical time encoding — midnight and 11pm are numerically close, unlike raw hour values |
| ⚡ Velocity | `tx_count_1h`, `tx_count_24h` | How many transactions happened in the preceding hour/day — a classic fraud tell |
| 🚨 Anomaly | `isolation_score` | How "weird" this transaction looks to a model that's only ever seen legitimate behavior |

---

## 🔬 Explainability

Predictions aren't a black box. Every scored transaction gets a **SHAP TreeExplainer** breakdown showing the top contributing features, their direction (pushing toward fraud or safety), and their exact magnitude — built for audit trails, not just accuracy leaderboards.

---

## ⚙️ Tech stack

| | |
|---|---|
| 🐍 Language | Python 3.12 |
| 🧮 ML core | scikit-learn, XGBoost, imbalanced-learn |
| 🔍 Explainability | SHAP |
| 📈 Visualization | Matplotlib |
| 🖥️ Interface | Streamlit |
| 💾 Serialization | joblib |

---

## 📁 Under the hood

```
src/
├── features.py   🧬  engineered feature pipeline
├── anomaly.py    🚨  isolation forest anomaly scorer
├── train.py      🎓  training + cross-validation + model export
└── explain.py    🔬  SHAP explainability utilities
streamlit_app.py  🖥️  the live scoring interface
```

Built as a self-contained, cloud-free deployment — no external API dependency, no per-request cost, model artifacts loaded straight from disk.
