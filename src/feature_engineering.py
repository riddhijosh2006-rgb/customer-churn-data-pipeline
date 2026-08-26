import pandas as pd

from sklearn.model_selection import train_test_split

# NOTE: This script demonstrates the train/test split step in
# isolation. The analytics-only engineered features (TenureGroup,
# ServiceCount, segments, etc, used by the dashboard) now live in
# src/pipeline/data_transformation.py (engineer_features()). Those
# analytics features are NOT fed to the ML model -- see that file's
# docstring for why.


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


print("\nFEATURES (X)")
print("-" * 40)
print(X.head())

print("\nTARGET (y)")
print("-" * 40)
print(y.head())


# ============================================================
# 3. IDENTIFY NUMERICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNUMERICAL FEATURES")
print("-" * 40)

for feature in numerical_features:
    print(feature)


# ============================================================
# 4. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()

print("\nCATEGORICAL FEATURES")
print("-" * 40)

for feature in categorical_features:
    print(feature)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTRAIN / TEST SPLIT")
print("-" * 40)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 6. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTRAINING TARGET DISTRIBUTION")
print("-" * 40)

print(
    y_train.value_counts(normalize=True) * 100
)


print("\nTEST TARGET DISTRIBUTION")
print("-" * 40)

print(
    y_test.value_counts(normalize=True) * 100
)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING SETUP COMPLETED")
print("=" * 60)