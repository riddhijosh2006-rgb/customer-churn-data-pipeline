"""
Data Ingestion
==============
Step 1 of the data pipeline.

Loads the raw Telco churn CSV, checks that the file actually exists,
logs basic facts about what got loaded, and hands back a DataFrame.
Nothing here cleans or transforms the data -- that happens later in
the pipeline.
"""

import logging
import os

import pandas as pd

from src.pipeline import config

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Raised when the raw dataset cannot be loaded."""


def _resolve_raw_path(path: str | None = None) -> str:
    """
    Figure out which raw CSV to read.

    Checks, in order: an explicit path argument, the new
    data/raw/ location, then the legacy data/ location (for
    backward compatibility with the original project layout).
    """
    if path:
        if not os.path.exists(path):
            raise DataIngestionError(f"Raw dataset not found at: {path}")
        return path

    if os.path.exists(config.RAW_DATA_PATH):
        return config.RAW_DATA_PATH

    if os.path.exists(config.LEGACY_RAW_DATA_PATH):
        logger.warning(
            "Using legacy raw data path %s -- consider moving the CSV "
            "into data/raw/",
            config.LEGACY_RAW_DATA_PATH,
        )
        return config.LEGACY_RAW_DATA_PATH

    raise DataIngestionError(
        "Raw dataset not found. Expected it at "
        f"'{config.RAW_DATA_PATH}' or '{config.LEGACY_RAW_DATA_PATH}'."
    )


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset.

    Parameters
    ----------
    path : str, optional
        Explicit path to a CSV file. If omitted, resolves the raw
        dataset path from config (data/raw/, falling back to data/).

    Returns
    -------
    pd.DataFrame
        The raw dataset, unmodified.

    Raises
    ------
    DataIngestionError
        If the file cannot be found or fails to parse.
    """
    resolved_path = _resolve_raw_path(path)

    try:
        df = pd.read_csv(resolved_path)
    except Exception as exc:  # noqa: BLE001 - want to wrap any parse error
        raise DataIngestionError(
            f"Failed to read CSV at '{resolved_path}': {exc}"
        ) from exc

    if df.empty:
        raise DataIngestionError(f"Loaded dataset is empty: '{resolved_path}'")

    logger.info("Loaded raw dataset from %s", resolved_path)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Dtypes:\n%s", df.dtypes)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_data()
    print("Ingested shape:", data.shape)
