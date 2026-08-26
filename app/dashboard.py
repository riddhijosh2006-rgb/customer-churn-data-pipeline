"""
Customer Churn Analysis Dashboard pages.

Every number shown here is calculated live from
data/processed/telco_churn_features.csv (or, if that hasn't been
generated yet, cleaned on the fly) -- nothing is hardcoded.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import config
from src.pipeline.data_ingestion import load_data, DataIngestionError
from src.pipeline.data_transformation import clean_data, engineer_features
from src.pipeline.data_validation import validate_data
from src.analytics import churn_analysis as ca
from src.analytics import customer_analysis as cu
from src.analytics import insights as ins


# ------------------------------------------------------------------
# DATA LOADING (cached)
# ------------------------------------------------------------------

@st.cache_data
def get_data() -> pd.DataFrame:
    """
    Load the analytics-ready dataset (with engineered features).

    Prefers the pre-computed file the data pipeline produces; falls
    back to running cleaning + feature engineering in-memory if the
    pipeline hasn't been run yet, so the dashboard never hard-fails.
    """
    if os.path.exists(config.PROCESSED_ANALYTICS_PATH):
        return pd.read_csv(config.PROCESSED_ANALYTICS_PATH)

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    return engineer_features(cleaned_df)


@st.cache_data
def get_data_quality_snapshot() -> dict:
    """Recompute a validation snapshot against the current raw data for the Data Quality page."""
    try:
        raw_df = load_data()
    except DataIngestionError as exc:
        return {"error": str(exc)}
    return {"raw_df": raw_df, "validation_report": validate_data(raw_df)}


# ------------------------------------------------------------------
# SHARED FILTER SIDEBAR (Customer Analysis page)
# ------------------------------------------------------------------

def _filter_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.subheader("Filters")
    filters = {}
    filters["Contract"] = st.sidebar.selectbox("Contract", ["All"] + sorted(df["Contract"].unique()))
    filters["InternetService"] = st.sidebar.selectbox("Internet Service", ["All"] + sorted(df["InternetService"].unique()))
    filters["gender"] = st.sidebar.selectbox("Gender", ["All"] + sorted(df["gender"].unique()))
    filters["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", ["All"] + sorted(df["SeniorCitizen"].unique().tolist()))
    filters["PaymentMethod"] = st.sidebar.selectbox("Payment Method", ["All"] + sorted(df["PaymentMethod"].unique()))
    filters["Churn"] = st.sidebar.selectbox("Churn", ["All"] + sorted(df["Churn"].unique()))
    return filters


# ------------------------------------------------------------------
# PAGE: OVERVIEW
# ------------------------------------------------------------------

def render_overview(df: pd.DataFrame):
    st.title("🏠 Overview")

    overall = ca.overall_churn(df)
    total_customers = len(df)
    high_risk = int(df["IsMonthToMonth"].sum()) if "IsMonthToMonth" in df.columns else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{overall['churned']:,}")
    col3.metric("Churn Rate", f"{overall['churn_rate']:.1f}%")
    col4.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
    col6.metric("Total Revenue (Charges)", f"${df['TotalCharges'].sum():,.0f}")
    if high_risk is not None:
        col7.metric("High-Risk Customers", f"{high_risk:,}", help="Customers on a month-to-month contract")

    st.divider()
    st.subheader("🔑 Key Insights")
    for line in ins.generate_insights(df):
        st.markdown(f"- {line}")


# ------------------------------------------------------------------
# PAGE: DATA ANALYSIS
# ------------------------------------------------------------------

def render_data_analysis(df: pd.DataFrame):
    st.title("📊 Data Analysis")

    st.subheader("Dataset Information")
    numeric_cols = df.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    col5, col6 = st.columns(2)
    col5.metric("Numerical Features", len(numeric_cols))
    col6.metric("Categorical Features", len(categorical_cols))

    st.divider()
    st.subheader("Distributions")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="Churn", title="Churn Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(df, names="gender", title="Gender Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(cu.distribution(df, "Contract"), x="Contract", y="Count", title="Contract Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.bar(cu.distribution(df, "InternetService"), x="InternetService", y="Count", title="Internet Service Distribution")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(cu.distribution(df, "PaymentMethod"), x="PaymentMethod", y="Count", title="Payment Method Distribution")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# PAGE: CUSTOMER ANALYSIS
# ------------------------------------------------------------------

def render_customer_analysis(df: pd.DataFrame):
    st.title("👥 Customer Analysis")

    filters = _filter_sidebar(df)
    filtered = cu.apply_filters(df, filters)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} customers after filters.")

    st.subheader("Demographics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(px.bar(cu.distribution(filtered, "gender"), x="gender", y="Count", title="Gender"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cu.distribution(filtered, "SeniorCitizen"), x="SeniorCitizen", y="Count", title="Senior Citizen"), use_container_width=True)
    with c3:
        st.plotly_chart(px.bar(cu.distribution(filtered, "Partner"), x="Partner", y="Count", title="Partner"), use_container_width=True)
    with c4:
        st.plotly_chart(px.bar(cu.distribution(filtered, "Dependents"), x="Dependents", y="Count", title="Dependents"), use_container_width=True)

    st.subheader("Customer Behavior")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.histogram(filtered, x="tenure", nbins=20, title="Tenure Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cu.distribution(filtered, "Contract"), x="Contract", y="Count", title="Contract"), use_container_width=True)
    with c3:
        st.plotly_chart(px.bar(cu.distribution(filtered, "InternetService"), x="InternetService", y="Count", title="Internet Service"), use_container_width=True)

    st.subheader("Services Subscribed")
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    service_counts = pd.DataFrame({
        "Service": service_cols,
        "Subscribed": [(filtered[c] == "Yes").sum() for c in service_cols],
    })
    st.plotly_chart(px.bar(service_counts, x="Service", y="Subscribed", title="Customers Subscribed per Service"), use_container_width=True)

    st.divider()
    st.subheader("🧩 Customer Segmentation")
    st.caption(
        "Tenure segments: New < 12 mo, Regular 12\u201336 mo, Long-Term \u2265 36 mo. "
        "Value segments: MonthlyCharges tertiles (Low/Medium/High), recomputed from the current data."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.bar(cu.segment_by_tenure(filtered), x="Segment", y="Customers", title="By Tenure"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cu.segment_by_value(filtered), x="Segment", y="Customers", title="By Value"), use_container_width=True)
    with c3:
        st.plotly_chart(px.bar(cu.segment_by_churn_risk(filtered), x="Segment", y="Customers", title="By Churn Risk (proxy)"), use_container_width=True)


# ------------------------------------------------------------------
# PAGE: CHURN ANALYSIS
# ------------------------------------------------------------------

def render_churn_analysis(df: pd.DataFrame):
    st.title("📈 Churn Analysis")

    overall = ca.overall_churn(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Count", f"{overall['churned']:,}")
    c2.metric("Non-Churn Count", f"{overall['retained']:,}")
    c3.metric("Churn Rate", f"{overall['churn_rate']:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names="Churn", title="Churn Distribution", hole=0.4), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "Contract"), x="Contract", y="ChurnRate", title="Churn Rate by Contract Type", text_auto=".1f"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "InternetService"), x="InternetService", y="ChurnRate", title="Churn Rate by Internet Service", text_auto=".1f"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "PaymentMethod"), x="PaymentMethod", y="ChurnRate", title="Churn Rate by Payment Method", text_auto=".1f"),
            use_container_width=True,
        )

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "TenureGroup"), x="TenureGroup", y="ChurnRate", title="Churn Rate by Tenure Group", text_auto=".1f"),
            use_container_width=True,
        )
    with c6:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "gender"), x="gender", y="ChurnRate", title="Churn Rate by Gender", text_auto=".1f"),
            use_container_width=True,
        )

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "SeniorCitizen"), x="SeniorCitizen", y="ChurnRate", title="Churn Rate by Senior Citizen", text_auto=".1f"),
            use_container_width=True,
        )
    with c8:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "MonthlyChargeGroup"), x="MonthlyChargeGroup", y="ChurnRate", title="Churn Rate by Monthly Charges", text_auto=".1f"),
            use_container_width=True,
        )

    c9, c10 = st.columns(2)
    with c9:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "TotalChargesGroup"), x="TotalChargesGroup", y="ChurnRate", title="Churn Rate by Total Charges", text_auto=".1f"),
            use_container_width=True,
        )
    with c10:
        st.plotly_chart(
            px.bar(ca.churn_rate_by(df, "ServiceCount"), x="ServiceCount", y="ChurnRate", title="Churn Rate by Additional Services Count", text_auto=".1f"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("🔑 Key Insights")
    for line in ins.generate_insights(df):
        st.markdown(f"- {line}")


# ------------------------------------------------------------------
# PAGE: REVENUE & CHARGES
# ------------------------------------------------------------------

def render_revenue(df: pd.DataFrame):
    st.title("💳 Revenue & Charges")

    contract_filter = st.selectbox("Filter by Contract", ["All"] + sorted(df["Contract"].unique()))
    filtered = df if contract_filter == "All" else df[df["Contract"] == contract_filter]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(filtered, x="MonthlyCharges", nbins=30, title="Monthly Charges Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.bar(ca.average_metric_by_churn(filtered, "MonthlyCharges"), x="Churn", y="AvgMonthlyCharges", title="Avg Monthly Charges by Churn Status"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            px.bar(ca.average_metric_by_churn(filtered, "TotalCharges"), x="Churn", y="AvgTotalCharges", title="Avg Total Charges by Churn Status"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.scatter(filtered, x="tenure", y="MonthlyCharges", color="Churn", title="Monthly Charges vs Tenure", opacity=0.5),
            use_container_width=True,
        )

    st.plotly_chart(
        px.box(filtered, x="Churn", y="TotalCharges", color="Churn", title="Total Charges by Churn Status"),
        use_container_width=True,
    )

    st.subheader("Revenue-Related Segments")
    st.caption("High Value = MonthlyCharges in the top 25% of the current dataset.")
    hv = filtered["HighValueCustomer"].value_counts().rename({True: "High Value", False: "Standard"}).reset_index()
    hv.columns = ["Segment", "Customers"]
    st.plotly_chart(px.bar(hv, x="Segment", y="Customers", title="High-Value vs Standard Customers"), use_container_width=True)


# ------------------------------------------------------------------
# PAGE: DATA QUALITY
# ------------------------------------------------------------------

def render_data_quality(df: pd.DataFrame):
    st.title("ℹ️ Data Quality")

    snapshot = get_data_quality_snapshot()
    if "error" in snapshot:
        st.error(f"Could not load raw data for validation: {snapshot['error']}")
        return

    report = snapshot["validation_report"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Shape", f"{report['n_rows']:,} x {report['n_columns']}")
    c2.metric("Missing Values", sum(report["missing_values"].values()))
    c3.metric("Duplicate Rows (raw)", report["duplicate_rows"])

    st.metric("Validation Status", "✅ Passed" if report["is_valid"] else "❌ Issues Found")

    if report["issues"]:
        st.error("Issues:\n" + "\n".join(f"- {i}" for i in report["issues"]))
    if report["warnings"]:
        st.warning("Warnings:\n" + "\n".join(f"- {w}" for w in report["warnings"]))

    st.divider()
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

    st.subheader("Column Statistics")
    # Cast to string: describe(include="all") mixes numeric and boolean/text
    # stats in the same columns, which pyarrow (Streamlit's table backend)
    # can't serialize directly.
    st.dataframe(df.describe(include="all").astype(str).transpose(), use_container_width=True)

    st.subheader("Target Distribution")
    st.dataframe(df["Churn"].value_counts().rename("Count"), use_container_width=True)

    st.divider()
    if os.path.exists(config.PIPELINE_RUN_LOG_PATH):
        import json
        with open(config.PIPELINE_RUN_LOG_PATH, "r", encoding="utf-8") as f:
            run_log = json.load(f)
        st.subheader("Last Pipeline Run")
        st.json(run_log)
    else:
        st.info("No pipeline run log found yet. Run `python -m src.pipeline.data_pipeline` to generate one.")
