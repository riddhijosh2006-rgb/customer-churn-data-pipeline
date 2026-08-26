"""
Data Pipeline - single entry point
====================================
Runs the full data pipeline end to end:

    Raw CSV -> Ingest -> Validate -> Clean -> Feature Engineer
    -> Save processed data -> Generate data quality report

Usage
-----
From the project root:

    python -m src.pipeline.data_pipeline

or, from other code:

    from src.pipeline.data_pipeline import run_data_pipeline
    result = run_data_pipeline()
"""

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from src.pipeline import config
from src.pipeline.data_ingestion import load_data
from src.pipeline.data_validation import validate_data
from src.pipeline.data_transformation import clean_data, engineer_features

logger = logging.getLogger(__name__)


def _build_data_quality_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                validation_report: dict) -> pd.DataFrame:
    """Build a flat, one-row-per-metric data quality report as a DataFrame."""

    numeric_cols = cleaned_df.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_cols = [c for c in cleaned_df.columns if c not in numeric_cols]

    # Duplicate rows are only meaningful once the non-informative
    # customerID column is dropped (every raw row has a unique ID).
    id_col = config.ID_COLUMN
    raw_without_id = raw_df.drop(columns=[id_col]) if id_col in raw_df.columns else raw_df
    duplicate_rows_removed = int(raw_without_id.duplicated().sum())

    target_distribution = (
        cleaned_df[config.TARGET_COLUMN].value_counts().to_dict()
        if config.TARGET_COLUMN in cleaned_df.columns
        else {}
    )

    rows = [
        {"metric": "raw_rows", "value": raw_df.shape[0]},
        {"metric": "raw_columns", "value": raw_df.shape[1]},
        {"metric": "cleaned_rows", "value": cleaned_df.shape[0]},
        {"metric": "cleaned_columns", "value": cleaned_df.shape[1]},
        {"metric": "duplicate_rows_removed", "value": duplicate_rows_removed},
        {"metric": "missing_values_in_raw", "value": sum(validation_report["missing_values"].values())},
        {"metric": "numerical_columns", "value": len(numeric_cols)},
        {"metric": "categorical_columns", "value": len(categorical_cols)},
        {"metric": "target_churn_yes", "value": target_distribution.get("Yes", 0)},
        {"metric": "target_churn_no", "value": target_distribution.get("No", 0)},
        {"metric": "validation_passed", "value": validation_report["is_valid"]},
    ]

    return pd.DataFrame(rows)


def run_data_pipeline(raw_path: str | None = None) -> dict:
    """
    Run the complete data pipeline and return a summary dict.

    Returns
    -------
    dict with keys: cleaned_df, features_df, validation_report,
    quality_report_path, processed_path, features_path, run_at
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    logger.info("STEP 1/6: Ingesting raw data")
    raw_df = load_data(raw_path)

    logger.info("STEP 2/6: Validating raw data")
    validation_report = validate_data(raw_df)
    if not validation_report["is_valid"]:
        logger.warning(
            "Validation found issues, proceeding anyway (see report): %s",
            validation_report["issues"],
        )

    logger.info("STEP 3/6: Cleaning data")
    cleaned_df = clean_data(raw_df)

    logger.info("STEP 4/6: Engineering analytics features")
    features_df = engineer_features(cleaned_df)

    logger.info("STEP 5/6: Saving processed data")
    cleaned_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    features_df.to_csv(config.PROCESSED_ANALYTICS_PATH, index=False)
    logger.info("Saved cleaned data to %s", config.PROCESSED_DATA_PATH)
    logger.info("Saved analytics features to %s", config.PROCESSED_ANALYTICS_PATH)

    logger.info("STEP 6/6: Generating data quality report")
    quality_report_df = _build_data_quality_report(raw_df, cleaned_df, validation_report)
    quality_report_df.to_csv(config.DATA_QUALITY_REPORT_PATH, index=False)

    run_at = datetime.now(timezone.utc).isoformat()
    run_log = {
        "run_at": run_at,
        "raw_rows": int(raw_df.shape[0]),
        "cleaned_rows": int(cleaned_df.shape[0]),
        "validation_passed": validation_report["is_valid"],
        "validation_issues": validation_report["issues"],
        "validation_warnings": validation_report["warnings"],
    }
    with open(config.PIPELINE_RUN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    logger.info("Data pipeline completed successfully.")

    return {
        "cleaned_df": cleaned_df,
        "features_df": features_df,
        "validation_report": validation_report,
        "quality_report_path": config.DATA_QUALITY_REPORT_PATH,
        "processed_path": config.PROCESSED_DATA_PATH,
        "features_path": config.PROCESSED_ANALYTICS_PATH,
        "run_at": run_at,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_data_pipeline()
    print("\nPipeline finished.")
    print("Cleaned rows:", summary["cleaned_df"].shape[0])
    print("Validation passed:", summary["validation_report"]["is_valid"])
    print("Quality report saved to:", summary["quality_report_path"])
