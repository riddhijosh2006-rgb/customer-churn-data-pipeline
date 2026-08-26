# 📊 Customer Churn Analytics & ML Pipeline

An end-to-end platform for telecom customer churn that combines **Data
Engineering**, **Data Analytics**, **Machine Learning**, and an
**interactive dashboard** for prediction — not just a trained model.

---

## 🚀 Overview

The project takes the raw **Telco Customer Churn dataset** and runs it
through a reusable data pipeline, an analytics layer, and a machine
learning pipeline, then surfaces all three through a single Streamlit
application:

```text
Data Engineering  →  Data Analytics  →  Machine Learning  →  Deployment
```

## 🎯 Problem Statement

Given information about a telecom customer — demographics, tenure,
internet/phone services, contract type, payment method, and billing —
predict whether the customer will churn (`Yes`) or stay (`No`). This
is a binary classification problem.

---

## 🏗️ Architecture

```text
                        RAW DATA
                           │
                           ▼
                    DATA INGESTION
                           │
                           ▼
                   DATA VALIDATION
                           │
                           ▼
                    DATA CLEANING
                           │
                           ▼
                 DATA TRANSFORMATION
                           │
                           ▼
                  FEATURE ENGINEERING
                           │
                           ▼
                    PROCESSED DATA
                       /       \
                      /         \
                     ▼           ▼
             ANALYTICS LAYER   ML PIPELINE
                     │              │
                     ▼              ▼
               DASHBOARD      MODEL TRAINING
                                    │
                                    ▼
                              SAVED MODEL
                                    │
                                    ▼
                            ML PREDICTION
```

**Data Pipeline** and **ML Pipeline** are kept deliberately separate:
the data pipeline produces two outputs — a model-ready cleaned dataset
(exact columns the trained model expects) and a separate,
analytics-only feature set (with extra engineered columns for the
dashboard). The analytics features are never fed into the ML model.

---

## 🧵 Data Pipeline

Location: `src/pipeline/`

| Module | Responsibility |
|---|---|
| `config.py` | Central paths, schema, feature-column list — single source of truth |
| `data_ingestion.py` | Loads the raw CSV, checks it exists, logs shape/columns/dtypes |
| `data_validation.py` | Checks required columns, duplicates, missing values, invalid types, unexpected categories, target distribution — returns a structured report instead of failing silently |
| `data_transformation.py` | `clean_data()` (blank `TotalCharges` → numeric, drop `customerID`, drop duplicate rows) and `engineer_features()` (adds dashboard-only analytics columns) |
| `data_pipeline.py` | `run_data_pipeline()` — single entry point that chains all of the above and writes outputs |

Run the full pipeline from the project root:

```bash
python -m src.pipeline.data_pipeline
```

This writes:

- `data/processed/cleaned_telco_churn.csv` — the model-ready dataset
- `data/processed/telco_churn_features.csv` — cleaned data + analytics features (TenureGroup, MonthlyChargeGroup, TotalChargesGroup, ServiceCount, IsLongTermCustomer, IsMonthToMonth, HighValueCustomer)
- `reports/data_quality_report.csv` — row/column counts, duplicates removed, missing values, target distribution, validation status
- `reports/last_pipeline_run.json` — timestamp and validation summary of the most recent run

The raw CSV can live at either `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
(preferred) or the original `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
location — ingestion checks both, so nothing breaks if you don't move
the file.

---

## 📊 Analytics Layer

Location: `src/analytics/`

| Module | Responsibility |
|---|---|
| `churn_analysis.py` | Overall churn rate, churn rate by category/numeric bucket, average metric by churn status |
| `customer_analysis.py` | Filtering, distributions, and three documented customer segmentations (tenure, value, churn-risk) |
| `insights.py` | `generate_insights(df)` — computes the dashboard's "Key Insights" text live from the data, nothing hardcoded |

---

## 🖥️ Dashboard

Location: `app/dashboard.py` (analytics pages) + `app/prediction.py` (ML page), wired together by `app/app.py`.

Sidebar-navigated pages:

- **🏠 Overview** — KPI cards (customers, churn rate, avg charges, avg tenure, revenue, high-risk count) + Key Insights
- **📊 Data Analysis** — dataset shape/missing/duplicate stats, churn/gender/contract/internet/payment distributions
- **👥 Customer Analysis** — demographic, behavioral, and service breakdowns with interactive filters (Contract, Internet Service, Gender, Senior Citizen, Payment Method, Churn), plus the three customer segmentations
- **📈 Churn Analysis** — 10 churn-rate breakdowns (contract, internet service, payment method, tenure group, gender, senior citizen, monthly/total charges, service count) — rates, not just raw counts
- **💳 Revenue & Charges** — monthly charges distribution, avg charges by churn status, charges vs tenure scatter, high-value vs standard segment
- **🤖 ML Prediction** — the original prediction form, unchanged behavior, now also showing a Risk Level (Low / Medium / High)
- **ℹ️ Data Quality** — live validation report, dtypes, column statistics, target distribution, and the last pipeline run log

All charts and KPIs are computed from the processed dataset at
render time via `@st.cache_data` — nothing is a hardcoded number.

### Segmentation rules (documented, not arbitrary)

- **Tenure**: New < 12 months, Regular 12–36 months, Long-Term ≥ 36 months
- **Value**: MonthlyCharges tertiles (Low / Medium / High), recomputed from the current data
- **Churn risk**: uses the ML model's probability where available (< 30% Low, 30–60% Medium, ≥ 60% High); otherwise a documented contract+tenure proxy rule

---

## 🤖 Machine Learning

The existing ML pipeline (`src/train.py`, `src/model_comparison.py`,
`src/hyperparameter_tuning.py`, `src/train_final.py`) is unchanged.
Workflow:

```text
Processed Data → Train/Test Split → Preprocessing (StandardScaler +
OneHotEncoder) → Model Comparison (Logistic Regression / Random
Forest / Gradient Boosting) → Hyperparameter Tuning (GridSearchCV) →
Final Model (tuned Logistic Regression) → Saved Pipeline
(models/churn_pipeline.joblib) → ML Prediction
```

The trained pipeline expects exactly these 19 raw columns (see
`src/pipeline/config.py::MODEL_FEATURE_COLUMNS`): `gender`,
`SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`,
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`,
`TotalCharges`. The dashboard's extra analytics columns are never
passed to it.

To retrain (optional — not required to run the dashboard):

```bash
python src/model_comparison.py
python src/hyperparameter_tuning.py
python src/train_final.py
```

---

## 📂 Project Structure

```text
customer-churn-ml-pipeline/
│
├── app/
│   ├── app.py              # Streamlit entry point, sidebar navigation
│   ├── dashboard.py         # Analytics pages (Overview, Data Analysis, etc.)
│   └── prediction.py        # ML Prediction page
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # legacy location, still supported
│   └── processed/
│       ├── cleaned_telco_churn.csv            # model-ready
│       └── telco_churn_features.csv           # + analytics features
│
├── models/
│   └── churn_pipeline.joblib
│
├── reports/
│   ├── data_quality_report.csv
│   └── last_pipeline_run.json
│
├── src/
│   ├── pipeline/
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── data_pipeline.py
│   │
│   ├── analytics/
│   │   ├── churn_analysis.py
│   │   ├── customer_analysis.py
│   │   └── insights.py
│   │
│   ├── data_cleaning.py         # standalone reference script
│   ├── data_inspection.py
│   ├── eda.py
│   ├── feature_engineering.py   # standalone reference script
│   ├── preprocessing.py
│   ├── model_comparison.py
│   ├── hyperparameter_tuning.py
│   ├── train.py
│   └── train_final.py
│
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline.data_pipeline
streamlit run app/app.py
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline.data_pipeline
streamlit run app/app.py
```

Running the data pipeline first is recommended (it generates the
analytics feature file and the data-quality report), but the
dashboard will also compute analytics features on the fly if you
skip that step.

---

## 🧰 Technologies

Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · Joblib

---

## ✅ Testing Performed

- `python -m src.pipeline.data_pipeline` run end-to-end: raw 7,043 rows → 7,021 cleaned rows (22 exact duplicate rows removed), quality report and run log generated correctly.
- Confirmed the existing `models/churn_pipeline.joblib` predicts correctly on the newly cleaned data (no column mismatch).
- All 7 dashboard pages exercised headlessly via Streamlit's `AppTest` harness — zero exceptions, including clicking "Predict Churn" and getting a probability + risk level back.
- `src/train_final.py` re-run successfully against the new cleaned dataset (produces an equivalent pipeline); the original shipped `churn_pipeline.joblib` was restored afterward so the shipped model is unchanged, per the "don't replace the model unnecessarily" requirement.
- All new/modified `.py` files pass `python -m py_compile`.

## ⚠️ Known Issues / Remaining Items

- Streamlit's `use_container_width` parameter is deprecated in newer Streamlit versions (still functional, shows a warning) — cosmetic only, not a functional issue.
- `src/data_cleaning.py` and `src/feature_engineering.py` are kept as standalone reference scripts and now carry a comment pointing to their reusable counterparts in `src/pipeline/`, rather than being deleted, to avoid breaking anyone used to running them directly.
