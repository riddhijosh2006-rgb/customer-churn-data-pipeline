"""
Central configuration for the data pipeline.

All paths are resolved relative to the project root so the pipeline
works no matter where it is launched from (as long as it's run from
the project root, e.g. `python -m src.pipeline.data_pipeline`).
"""

import os

# ------------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------------
# config.py lives at <root>/src/pipeline/config.py -> go up 3 levels
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# ------------------------------------------------------------------
# DATA PATHS
# ------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Preferred (new) location, with a fallback to the legacy flat location
# so nothing breaks if someone still points at data/<file>.csv
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILENAME)
LEGACY_RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_FILENAME)

PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, "cleaned_telco_churn.csv"
)

# Full feature set (with engineered analytics columns), used by the
# dashboard's analytics pages only -- NEVER fed to the ML model.
PROCESSED_ANALYTICS_PATH = os.path.join(
    PROCESSED_DATA_DIR, "telco_churn_features.csv"
)

# ------------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_pipeline.joblib")

# ------------------------------------------------------------------
# REPORTS
# ------------------------------------------------------------------
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_QUALITY_REPORT_PATH = os.path.join(
    REPORTS_DIR, "data_quality_report.csv"
)
PIPELINE_RUN_LOG_PATH = os.path.join(
    REPORTS_DIR, "last_pipeline_run.json"
)

# ------------------------------------------------------------------
# SCHEMA
# ------------------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Exact columns the trained churn_pipeline.joblib model was fit on.
# This MUST stay in sync with the model -- do not add engineered
# features to this list without retraining the model.
MODEL_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

REQUIRED_COLUMNS = [ID_COLUMN] + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    c for c in MODEL_FEATURE_COLUMNS if c not in NUMERIC_COLUMNS
]
