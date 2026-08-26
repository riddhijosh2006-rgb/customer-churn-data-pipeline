"""
Churn Analysis
==============
Pure, reusable functions that compute churn-rate breakdowns from the
processed (features) DataFrame. Every function here returns
plain pandas objects -- no Streamlit calls -- so they're easy to unit
test and reuse from the dashboard.
"""

import pandas as pd


def overall_churn(df: pd.DataFrame) -> dict:
    """Total churned/retained counts and overall churn rate (%)."""
    counts = df["Churn"].value_counts()
    churned = int(counts.get("Yes", 0))
    retained = int(counts.get("No", 0))
    total = churned + retained
    rate = (churned / total * 100) if total else 0.0
    return {"churned": churned, "retained": retained, "churn_rate": rate}


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Churn rate (%) broken down by a categorical column, e.g.
    'Contract', 'InternetService', 'PaymentMethod', 'TenureGroup'.

    Returns a DataFrame with columns: [column, "ChurnRate", "Customers"].
    """
    grouped = (
        df.groupby(column)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )
    counts = df.groupby(column).size().reset_index(name="Customers")
    result = grouped.merge(counts, on=column)
    return result.sort_values("ChurnRate", ascending=False)


def churn_rate_by_numeric_bucket(df: pd.DataFrame, column: str, bins: int = 5) -> pd.DataFrame:
    """
    Churn rate (%) across quantile buckets of a numeric column, e.g.
    'MonthlyCharges' or 'TotalCharges'.
    """
    bucketed = pd.qcut(df[column], q=bins, duplicates="drop")
    temp = df.assign(_bucket=bucketed.astype(str))
    result = churn_rate_by(temp, "_bucket").rename(columns={"_bucket": column})
    return result


def average_metric_by_churn(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Average of a numeric column (e.g. tenure, MonthlyCharges) grouped by churn status."""
    return df.groupby("Churn")[column].mean().reset_index(name=f"Avg{column}")
