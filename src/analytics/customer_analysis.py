"""
Customer Analysis
=================
Reusable helpers for demographic / behavioral / service breakdowns and
customer segmentation used by the "Customer Analysis" dashboard page.
"""

import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply a dict of {column: selected_value_or_'All'} filters.
    Any filter whose value is "All" (or empty/None) is skipped.
    """
    filtered = df
    for column, value in filters.items():
        if value in (None, "All", [], ()):
            continue
        if isinstance(value, (list, tuple, set)):
            filtered = filtered[filtered[column].isin(value)]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered


def distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Value counts for a categorical column as a tidy DataFrame: [column, 'Count']."""
    return df[column].value_counts().reset_index().rename(columns={"count": "Count"})


def segment_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by tenure into New / Regular / Long-Term.

    Rule (documented for the dashboard):
      New Customers     : tenure < 12 months
      Regular Customers : 12 <= tenure < 36 months
      Long-Term Customers: tenure >= 36 months
    """
    def _label(t):
        if t < 12:
            return "New Customers"
        if t < 36:
            return "Regular Customers"
        return "Long-Term Customers"

    segment = df["tenure"].apply(_label)
    counts = segment.value_counts().reindex(
        ["New Customers", "Regular Customers", "Long-Term Customers"]
    ).fillna(0).astype(int)
    return counts.rename_axis("Segment").reset_index(name="Customers")


def segment_by_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by value using MonthlyCharges tertiles
    (data-driven cutoffs, recomputed from the current dataset):
      Low Value    : bottom third of MonthlyCharges
      Medium Value : middle third
      High Value   : top third
    """
    labels = pd.qcut(
        df["MonthlyCharges"], q=3, labels=["Low Value", "Medium Value", "High Value"],
        duplicates="drop",
    )
    counts = labels.value_counts().reindex(
        ["Low Value", "Medium Value", "High Value"]
    ).fillna(0).astype(int)
    return counts.rename_axis("Segment").reset_index(name="Customers")


def segment_by_churn_risk(df: pd.DataFrame, churn_probabilities=None) -> pd.DataFrame:
    """
    Segment customers by churn risk.

    If per-row model churn probabilities are supplied, buckets them
    into Low/Medium/High using the same thresholds as the single
    ML Prediction page (<30% Low, 30-60% Medium, >=60% High).

    If probabilities are NOT supplied (no model scores available),
    falls back to a simple, clearly-labelled proxy rule based on
    contract type and tenure:
      High Risk   : Month-to-month contract AND tenure < 12 months
      Medium Risk : Month-to-month contract, tenure >= 12 months
      Low Risk    : One year / Two year contract
    """
    if churn_probabilities is not None:
        def _bucket(p):
            if p >= 0.60:
                return "High Churn Risk"
            if p >= 0.30:
                return "Medium Churn Risk"
            return "Low Churn Risk"

        labels = churn_probabilities.apply(_bucket)
    else:
        def _proxy(row):
            if row["Contract"] == "Month-to-month" and row["tenure"] < 12:
                return "High Churn Risk"
            if row["Contract"] == "Month-to-month":
                return "Medium Churn Risk"
            return "Low Churn Risk"

        labels = df.apply(_proxy, axis=1)

    counts = labels.value_counts().reindex(
        ["Low Churn Risk", "Medium Churn Risk", "High Churn Risk"]
    ).fillna(0).astype(int)
    return counts.rename_axis("Segment").reset_index(name="Customers")
