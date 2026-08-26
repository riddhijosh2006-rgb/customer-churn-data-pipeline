import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)

print("=" * 60)
print("CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ============================================================
# 3. REMOVE CUSTOMER ID
# ============================================================

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


# ============================================================
# 6. CREATE PREPROCESSOR
# ============================================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

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


# ============================================================
# 7. CREATE ML PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


print("\nML Pipeline:")
print(model_pipeline)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model_pipeline.predict(X_test)


# ============================================================
# 10. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMODEL ACCURACY")
print("-" * 40)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("-" * 40)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("-" * 40)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

