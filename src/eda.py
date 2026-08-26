import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)

print("=" * 60)
print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")
print("-" * 40)
print(df.shape)


print("\n2. FIRST 5 ROWS")
print("-" * 40)
print(df.head())


print("\n3. DATA TYPES")
print("-" * 40)
print(df.dtypes)


# ============================================================
# 3. NUMERICAL SUMMARY
# ============================================================

print("\n4. NUMERICAL STATISTICS")
print("-" * 40)
print(df.describe())


# ============================================================
# 4. CATEGORICAL COLUMN ANALYSIS
# ============================================================

print("\n5. CATEGORICAL FEATURES")
print("-" * 40)

categorical_columns = df.select_dtypes(
    include="str"
).columns

for column in categorical_columns:

    print(f"\n{column}")
    print(df[column].value_counts())


# ============================================================
# 5. CHURN DISTRIBUTION
# ============================================================

print("\n6. CHURN DISTRIBUTION")
print("-" * 40)

churn_counts = df["Churn"].value_counts()

print(churn_counts)

print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ============================================================
# 6. CONTRACT VS CHURN
# ============================================================

print("\n7. CONTRACT VS CHURN")
print("-" * 40)

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)


# ============================================================
# 7. TENURE VS CHURN
# ============================================================

print("\n8. AVERAGE TENURE BY CHURN")
print("-" * 40)

tenure_churn = df.groupby(
    "Churn"
)["tenure"].mean()

print(tenure_churn)


# ============================================================
# 8. MONTHLY CHARGES VS CHURN
# ============================================================

print("\n9. AVERAGE MONTHLY CHARGES BY CHURN")
print("-" * 40)

monthly_charges_churn = df.groupby(
    "Churn"
)["MonthlyCharges"].mean()

print(monthly_charges_churn)


# ============================================================
# 9. TOTAL CHARGES VS CHURN
# ============================================================

print("\n10. AVERAGE TOTAL CHARGES BY CHURN")
print("-" * 40)

total_charges_churn = df.groupby(
    "Churn"
)["TotalCharges"].mean()

print(total_charges_churn)


# ============================================================
# 10. INTERNET SERVICE VS CHURN
# ============================================================

print("\n11. INTERNET SERVICE VS CHURN")
print("-" * 40)

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn)


# ============================================================
# 11. PAYMENT METHOD VS CHURN
# ============================================================

print("\n12. PAYMENT METHOD VS CHURN")
print("-" * 40)

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print(payment_churn)


# ============================================================
# 12. TECH SUPPORT VS CHURN
# ============================================================

print("\n13. TECH SUPPORT VS CHURN")
print("-" * 40)

tech_support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(tech_support_churn)


# ============================================================
# 13. PARTNER VS CHURN
# ============================================================

print("\n14. PARTNER VS CHURN")
print("-" * 40)

partner_churn = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print(partner_churn)


# ============================================================
# 14. DEPENDENTS VS CHURN
# ============================================================

print("\n15. DEPENDENTS VS CHURN")
print("-" * 40)

dependents_churn = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
) * 100

print(dependents_churn)


# ============================================================
#                    VISUALIZATIONS
# ============================================================


# ============================================================
# 15. CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 16. CONTRACT VS CHURN
# ============================================================

contract_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 17. TENURE VS CHURN
# ============================================================

df.boxplot(
    column="tenure",
    by="Churn",
    figsize=(7, 5)
)

plt.title("Tenure Distribution by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()


# ============================================================
# 18. MONTHLY CHARGES VS CHURN
# ============================================================

df.boxplot(
    column="MonthlyCharges",
    by="Churn",
    figsize=(7, 5)
)

plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ============================================================
# 19. TOTAL CHARGES VS CHURN
# ============================================================

df.boxplot(
    column="TotalCharges",
    by="Churn",
    figsize=(7, 5)
)

plt.title("Total Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.show()


# ============================================================
# 20. INTERNET SERVICE VS CHURN
# ============================================================

internet_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 21. PAYMENT METHOD VS CHURN
# ============================================================

payment_churn.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage")
plt.xticks(rotation=25)

plt.tight_layout()
plt.show()


# ============================================================
# 22. TECH SUPPORT VS CHURN
# ============================================================

tech_support_churn.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Churn Rate by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 23. PARTNER VS CHURN
# ============================================================

partner_churn.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Churn Rate by Partner")
plt.xlabel("Partner")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 24. DEPENDENTS VS CHURN
# ============================================================

dependents_churn.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Churn Rate by Dependents")
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)