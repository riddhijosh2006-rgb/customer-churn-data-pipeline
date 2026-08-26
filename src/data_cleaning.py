import pandas as pd

# NOTE: This standalone script is kept for reference / manual runs.
# The reusable, production version of this same cleaning logic now
# lives in src/pipeline/data_transformation.py (clean_data()), which
# is what src/pipeline/data_pipeline.py and the dashboard actually
# use. Run `python -m src.pipeline.data_pipeline` for the full,
# validated pipeline run.

# ==========================================
# 1. LOAD RAW DATA
# ==========================================

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Shape before cleaning:", df.shape)


# ==========================================
# 2. INSPECT EMPTY TOTAL CHARGES
# ==========================================

empty_total_charges = df[
    df["TotalCharges"].str.strip() == ""
]

print("\nEmpty TotalCharges rows:")
print(
    empty_total_charges[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)


# ==========================================
# 3. CLEAN TOTAL CHARGES
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Replace missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ==========================================
# 4. REMOVE CUSTOMER ID
# ==========================================

df = df.drop(columns=["customerID"])


# ==========================================
# 5. CHECK DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)


# ==========================================
# 6. CHECK MISSING VALUES
# ==========================================

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ==========================================
# 7. CHECK DATA TYPES
# ==========================================

print("\nData types after cleaning:")
print(df.dtypes)


# ==========================================
# 8. FINAL SHAPE
# ==========================================

print("\nShape after cleaning:", df.shape)


# ==========================================
# 9. SAVE CLEANED DATA
# ==========================================

df.to_csv(
    "data/processed/cleaned_telco_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")