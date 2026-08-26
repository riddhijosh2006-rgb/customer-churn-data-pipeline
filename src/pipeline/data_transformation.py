"""
Data Transformation
====================
Steps 3-4 of the data pipeline: cleaning and feature engineering.

This module intentionally produces TWO outputs:

1. `clean_data()` -> the cleaned dataset in the EXACT column format
   the existing trained model (`models/churn_pipeline.joblib`) was
   fit on. This is what gets saved to
   `data/processed/cleaned_telco_churn.csv` and is what the ML
   pipeline (train.py / train_final.py / app prediction) consumes.

2. `engineer_features()` -> takes the cleaned dataset and ADDS
   analytics-only columns (tenure buckets, service counts, value
   segments, etc). This is saved separately to
   `data/processed/telco_churn_features.csv` and is used ONLY by the
   dashboard's analytics pages -- never passed into the ML model,
   since the model was not trained on these columns (that would be a
   silent feature-mismatch bug).
"""

import logging

import pandas as pd

from src.pipeline import config

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco churn dataset.

    Mirrors the logic originally in src/data_cleaning.py:
    - Coerce blank/invalid TotalCharges to numeric (missing -> 0)
    - Drop the customerID identifier column
    - Drop exact duplicate rows

    Returns a DataFrame with the same 20 columns
    (19 model features + Churn) the trained model expects.
    """
    df = df.copy()

    before_shape = df.shape

    # --- TotalCharges: stored as text with some blank entries ---------
    # Checked with pd.api.types.is_numeric_dtype rather than `== object`
    # because pandas' newer string dtype ("str") is not `object`, but
    # still needs the same strip + coerce treatment.
    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- drop identifier column (not a useful ML feature) -----------------
    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])

    # --- drop exact duplicate rows ------------------------------------
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows", duplicate_count)

    logger.info("Cleaned data: %s -> %s", before_shape, df.shape)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analytics-only engineered features to the CLEANED dataset.

    IMPORTANT: the output of this function is for the dashboard's
    analytics/segmentation pages only. It must never be passed into
    `churn_pipeline.joblib`, which only knows about the original 19
    feature columns.

    Added columns:
    - TenureGroup        : bucketed tenure (0-12, 13-24, 25-48, 49-60, 61-72)
    - MonthlyChargeGroup  : bucketed MonthlyCharges (Low/Medium/High)
    - TotalChargesGroup   : bucketed TotalCharges (Low/Medium/High)
    - ServiceCount        : count of add-on services subscribed to (0-6)
    - IsLongTermCustomer  : tenure >= 24 months
    - IsMonthToMonth      : Contract == "Month-to-month"
    - HighValueCustomer   : MonthlyCharges in the top quartile
    """
    df = df.copy()

    # --- TenureGroup --------------------------------------------------
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    ).astype(str)

    # --- MonthlyChargeGroup (tertiles: data-driven, not hardcoded) --------
    df["MonthlyChargeGroup"] = pd.qcut(
        df["MonthlyCharges"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    ).astype(str)

    # --- TotalChargesGroup (tertiles) --------------------------------------
    df["TotalChargesGroup"] = pd.qcut(
        df["TotalCharges"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    ).astype(str)

    # --- ServiceCount: number of add-on services actively subscribed ------
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["ServiceCount"] = (df[service_columns] == "Yes").sum(axis=1)

    # --- Boolean/binary business flags -------------------------------------
    df["IsLongTermCustomer"] = df["tenure"] >= 24
    df["IsMonthToMonth"] = df["Contract"] == "Month-to-month"

    # HighValueCustomer: top quartile of MonthlyCharges (data-driven cutoff)
    high_value_cutoff = df["MonthlyCharges"].quantile(0.75)
    df["HighValueCustomer"] = df["MonthlyCharges"] >= high_value_cutoff

    logger.info("Feature engineering added columns: TenureGroup, "
                "MonthlyChargeGroup, TotalChargesGroup, ServiceCount, "
                "IsLongTermCustomer, IsMonthToMonth, HighValueCustomer")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.pipeline.data_ingestion import load_data

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    features_df = engineer_features(cleaned_df)
    print(features_df.head())
