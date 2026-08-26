import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
print("CUSTOMER CHURN - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# Remove customer ID if present
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
# 6. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_pipelines = {}


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"TRAINING: {model_name}")
    print("-" * 70)

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability predictions
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
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

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Store trained pipeline
    trained_pipelines[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 8. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 9. SORT BY F1 SCORE
# ============================================================

results_sorted = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODELS SORTED BY F1 SCORE")
print("=" * 70)

print(
    results_sorted.to_string(index=False)
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model_name = results_sorted.iloc[0]["Model"]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Model: {best_model_name}")


print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)