import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)

print("=" * 70)
print("CUSTOMER CHURN - HYPERPARAMETER TUNING")
print("=" * 70)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# Remove customer ID
if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)


# ============================================================
# 3. IDENTIFY FEATURE TYPES
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
# 6. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
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


# ============================================================
# 7. DEFINE HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


print("\nHyperparameter combinations:")
print(param_grid)


# ============================================================
# 8. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


print("\nStarting GridSearchCV...")
print("Please wait...")


grid_search.fit(
    X_train,
    y_train
)


print("\nGridSearchCV completed!")


# ============================================================
# 9. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(
    grid_search.best_params_
)


# ============================================================
# 10. BEST CROSS-VALIDATION SCORE
# ============================================================

print("\nBEST CROSS-VALIDATION F1 SCORE")
print("-" * 40)

print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 11. GET BEST PIPELINE
# ============================================================

best_pipeline = grid_search.best_estimator_


# ============================================================
# 12. TEST SET PREDICTIONS
# ============================================================

y_pred = best_pipeline.predict(
    X_test
)

y_probability = best_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. EVALUATE FINAL TUNED MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TUNED MODEL TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 15. COMPARISON WITH BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE VS TUNED MODEL")
print("=" * 70)

print("\nBaseline Logistic Regression:")
print("Accuracy : 0.8055")
print("Precision: 0.6572")
print("Recall   : 0.5588")
print("F1 Score : 0.6040")
print("ROC-AUC  : 0.8421")

print("\nTuned Logistic Regression:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)