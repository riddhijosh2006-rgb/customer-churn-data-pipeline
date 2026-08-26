import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


# ============================================================
# 3. REMOVE CUSTOMER ID
# ============================================================

# customerID is an identifier, not a useful ML feature.
if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()


print("=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


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


print("\nTrain/Test Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 6. NUMERICAL TRANSFORMER
# ============================================================

numeric_transformer = StandardScaler()


# ============================================================
# 7. CATEGORICAL TRANSFORMER
# ============================================================

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


# ============================================================
# 8. CREATE COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nColumnTransformer created successfully!")


# ============================================================
# 9. FIT TRANSFORMER ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)



# ============================================================
# 10. TRANSFORM TEST DATA
# ============================================================

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 11. CHECK PROCESSED DATA
# ============================================================

print("\nProcessed Data:")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape :", X_test_processed.shape)


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)