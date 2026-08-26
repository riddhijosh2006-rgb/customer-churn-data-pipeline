"""
Business Insights
==================
Generates plain-English insight statements FROM the actual processed
dataset -- every number here is calculated, never hardcoded. Used by
the dashboard's "Key Insights" section.
"""

import pandas as pd

from src.analytics.churn_analysis import churn_rate_by, overall_churn, average_metric_by_churn


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable insight strings computed from df."""
    insights: list[str] = []

    overall = overall_churn(df)
    insights.append(
        f"Overall churn rate is {overall['churn_rate']:.1f}% "
        f"({overall['churned']} of {overall['churned'] + overall['retained']} customers)."
    )

    # Contract type
    if "Contract" in df.columns:
        by_contract = churn_rate_by(df, "Contract")
        top = by_contract.iloc[0]
        bottom = by_contract.iloc[-1]
        insights.append(
            f"Customers on {top['Contract']} contracts churn the most "
            f"({top['ChurnRate']:.1f}%), vs {bottom['ChurnRate']:.1f}% "
            f"for {bottom['Contract']} contracts."
        )

    # Tenure
    if "tenure" in df.columns:
        tenure_by_churn = average_metric_by_churn(df, "tenure")
        churned_avg = tenure_by_churn.loc[tenure_by_churn["Churn"] == "Yes", "Avgtenure"]
        retained_avg = tenure_by_churn.loc[tenure_by_churn["Churn"] == "No", "Avgtenure"]
        if not churned_avg.empty and not retained_avg.empty:
            insights.append(
                f"Churned customers had an average tenure of {churned_avg.values[0]:.1f} months, "
                f"compared to {retained_avg.values[0]:.1f} months for retained customers."
            )

    # Monthly charges
    if "MonthlyCharges" in df.columns:
        mc_by_churn = average_metric_by_churn(df, "MonthlyCharges")
        churned_avg = mc_by_churn.loc[mc_by_churn["Churn"] == "Yes", "AvgMonthlyCharges"]
        retained_avg = mc_by_churn.loc[mc_by_churn["Churn"] == "No", "AvgMonthlyCharges"]
        if not churned_avg.empty and not retained_avg.empty and churned_avg.values[0] > retained_avg.values[0]:
            insights.append(
                f"Churned customers pay ${churned_avg.values[0]:.2f}/month on average, "
                f"higher than retained customers at ${retained_avg.values[0]:.2f}/month."
            )

    # Payment method
    if "PaymentMethod" in df.columns:
        by_payment = churn_rate_by(df, "PaymentMethod")
        top = by_payment.iloc[0]
        insights.append(
            f"{top['PaymentMethod']} has the highest churn rate among payment "
            f"methods, at {top['ChurnRate']:.1f}%."
        )

    # Internet service / tech support
    if "TechSupport" in df.columns:
        by_support = churn_rate_by(df, "TechSupport")
        no_support = by_support[by_support["TechSupport"] == "No"]
        yes_support = by_support[by_support["TechSupport"] == "Yes"]
        if not no_support.empty and not yes_support.empty:
            insights.append(
                f"Customers without Tech Support churn at {no_support['ChurnRate'].values[0]:.1f}%, "
                f"vs {yes_support['ChurnRate'].values[0]:.1f}% for those with it."
            )

    return insights
