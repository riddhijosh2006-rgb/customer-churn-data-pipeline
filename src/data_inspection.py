import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Display first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Data types
print("\nDATA TYPES")
print(df.dtypes)

# Dataset information
print("\nDATASET INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Target distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE")
print(df["Churn"].value_counts(normalize=True) * 100)
print("\nTOTAL CHARGES INVESTIGATION")

print("Empty strings:")
print(
    df["TotalCharges"]
    .str.strip()
    .eq("")
    .sum()
)

print("\nFirst 20 unique values:")
print(
    df["TotalCharges"]
    .str.strip()
    .unique()[:20]
)