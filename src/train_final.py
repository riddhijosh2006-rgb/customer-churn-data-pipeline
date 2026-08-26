import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# Remove customer ID
if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)


# ============================================================
# 3. FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 6. FINAL MODEL
# ============================================================

model = LogisticRegression(
    C=1,
    class_weight=None,
    max_iter=1000
)


# ============================================================
# 7. COMPLETE PIPELINE
# ============================================================

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 8. TRAIN
# ============================================================

print("=" * 60)
print("TRAINING FINAL CUSTOMER CHURN PIPELINE")
print("=" * 60)

final_pipeline.fit(
    X_train,
    y_train
)

print("Final model trained successfully!")


# ============================================================
# 9. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 10. SAVE COMPLETE PIPELINE
# ============================================================

model_path = "models/churn_pipeline.joblib"

joblib.dump(
    final_pipeline,
    model_path
)


print("\nModel saved successfully!")
print(f"Location: {model_path}")

print("\n" + "=" * 60)
print("FINAL PIPELINE READY")
print("=" * 60)