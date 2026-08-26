"""
Customer Churn Analytics & ML Pipeline - Streamlit Application

Entry point. Wires up sidebar navigation across the dashboard pages
(analytics, powered by app/dashboard.py) and the ML prediction page
(app/prediction.py). Run from the project root:

    streamlit run app/app.py
"""

import os
import sys

# Add both the project root (so `from src...` imports work) and this
# script's own directory (so `dashboard`/`prediction` import as plain
# top-level modules) to sys.path. We deliberately do NOT import them
# as `app.dashboard` / `app.prediction` -- a package named "app"
# containing a module also named "app.py" causes Python to treat the
# running script and the package as the same partially-initialized
# module, raising a circular-import error (seen on Windows).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
for p in (PROJECT_ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

import dashboard  # noqa: E402
import prediction  # noqa: E402


st.set_page_config(
    page_title="Customer Churn Analytics & ML Pipeline",
    page_icon="📊",
    layout="wide",
)

PAGES = {
    "🏠 Overview": dashboard.render_overview,
    "📊 Data Analysis": dashboard.render_data_analysis,
    "👥 Customer Analysis": dashboard.render_customer_analysis,
    "📈 Churn Analysis": dashboard.render_churn_analysis,
    "💳 Revenue & Charges": dashboard.render_revenue,
    "🤖 ML Prediction": None,  # handled separately, doesn't need `df`
    "ℹ️ Data Quality": dashboard.render_data_quality,
}


def main():
    st.sidebar.title("📊 Churn Analytics")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.divider()

    if page == "🤖 ML Prediction":
        prediction.render()
        return

    try:
        df = dashboard.get_data()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Could not load the processed dataset. Try running "
            "`python -m src.pipeline.data_pipeline` first.\n\n"
            f"Details: {exc}"
        )
        return

    PAGES[page](df)


if __name__ == "__main__":
    main()
