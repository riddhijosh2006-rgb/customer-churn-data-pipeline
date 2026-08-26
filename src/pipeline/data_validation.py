"""
Data Validation
================
Step 2 of the data pipeline.

Runs a battery of checks against the raw dataset BEFORE cleaning, and
returns a structured report instead of silently passing or crashing.
Cleaning/feature code can inspect `report["is_valid"]` and decide
whether to proceed.
"""

import logging

import pandas as pd

from src.pipeline import config

logger = logging.getLogger(__name__)

EXPECTED_CATEGORIES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the raw dataset and return a report dict. Never raises
    on data-quality issues -- it records them so the caller can log,
    warn, or halt as appropriate.
    """
    issues: list[str] = []
    warnings: list[str] = []

    # --- required columns -------------------------------------------------
    missing_columns = [
        c for c in config.REQUIRED_COLUMNS if c not in df.columns
    ]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    # --- duplicate rows ------------------------------------------------
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate rows found")

    # --- duplicate customer IDs -----------------------------------------
    if config.ID_COLUMN in df.columns:
        dup_ids = int(df[config.ID_COLUMN].duplicated().sum())
        if dup_ids:
            issues.append(f"{dup_ids} duplicate customerID values found")

    # --- missing values ---------------------------------------------------
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]
    if not missing_per_column.empty:
        warnings.append(f"Missing values found: {missing_per_column.to_dict()}")

    # --- TotalCharges: known Telco quirk (blank strings, not NaN) ---------
    if "TotalCharges" in df.columns and not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        blank_count = int(df["TotalCharges"].astype(str).str.strip().eq("").sum())
        if blank_count:
            warnings.append(
                f"{blank_count} blank TotalCharges values will be coerced"
            )

    # --- numeric columns should be numeric-ish -----------------------------
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column '{col}' expected numeric, got {df[col].dtype}")

    # --- target column ------------------------------------------------
    if config.TARGET_COLUMN in df.columns:
        target_values = set(df[config.TARGET_COLUMN].dropna().unique())
        unexpected_target = target_values - EXPECTED_CATEGORIES[config.TARGET_COLUMN]
        if unexpected_target:
            issues.append(f"Unexpected target values: {unexpected_target}")
    else:
        issues.append("Target column 'Churn' is missing")

    # --- unexpected categorical values ------------------------------------
    unexpected_categories: dict = {}
    for col, expected in EXPECTED_CATEGORIES.items():
        if col in df.columns:
            actual = set(df[col].dropna().unique())
            unexpected = actual - expected
            if unexpected:
                unexpected_categories[col] = list(unexpected)

    if unexpected_categories:
        warnings.append(f"Unexpected categorical values: {unexpected_categories}")

    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_count,
        "missing_values": missing_per_column.to_dict(),
        "unexpected_categories": unexpected_categories,
        "issues": issues,
        "warnings": warnings,
        "is_valid": len(issues) == 0,
    }

    logger.info("Validation report: %s", report)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.pipeline.data_ingestion import load_data

    raw_df = load_data()
    result = validate_data(raw_df)
    print(result)
