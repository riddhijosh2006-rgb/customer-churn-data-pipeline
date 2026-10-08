# 📊 Customer Churn Analytics & ML Pipeline

An end-to-end **Customer Churn Analytics and Machine Learning pipeline** that combines **Data Engineering, Data Analytics, Machine Learning, and interactive dashboard deployment** into a single reusable application.

The project processes raw telecom customer data, validates and transforms it, generates analytical insights, trains machine learning models, and provides an interactive Streamlit dashboard for churn analysis and prediction.

---

## 🚀 Project Overview

Customer churn is a major business problem for subscription-based companies.

The objective of this project is to:

- Clean and validate customer data
- Build a reusable data processing pipeline
- Analyze customer churn patterns
- Identify important customer segments
- Train and compare machine learning models
- Predict the probability of customer churn
- Classify customers into different risk levels
- Present the results through an interactive dashboard

### End-to-End Workflow

```text
Raw Customer Data
        ↓
Data Ingestion
        ↓
Data Validation
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Processed Data
        ↓
   ┌────┴────┐
   ↓         ↓
Analytics    ML Pipeline
   ↓         ↓
Dashboard   Model Training
              ↓
        Model Evaluation
              ↓
         Saved Pipeline
              ↓
        Churn Prediction
```

---

# 🎯 Problem Statement

Given information about a telecom customer such as:

- Demographics
- Tenure
- Contract type
- Internet and phone services
- Payment method
- Monthly charges
- Total charges

the system predicts whether the customer is likely to **churn (`Yes`) or stay (`No`)**.

This is treated as a **binary classification problem**.

---

# 🏗️ System Architecture

```text
                         RAW DATA
                            │
                            ▼
                    DATA INGESTION
                            │
                            ▼
                    DATA VALIDATION
                            │
                            ▼
                     DATA CLEANING
                            │
                            ▼
                  DATA TRANSFORMATION
                            │
                            ▼
                   FEATURE ENGINEERING
                            │
                            ▼
                    PROCESSED DATA
                       /          \
                      /            \
                     ▼              ▼
              ANALYTICS LAYER    ML PIPELINE
                     │              │
                     ▼              ▼
                DASHBOARD       MODEL TRAINING
                                    │
                                    ▼
                              MODEL EVALUATION
                                    │
                                    ▼
                               SAVED MODEL
                                    │
                                    ▼
                             CHURN PREDICTION
```

A key design decision is keeping the **analytics features separate from the model-ready features**.

The machine learning model receives only the columns it was trained on, while additional engineered features are used specifically for dashboard analytics.

---

# 🧵 Data Engineering Pipeline

The data pipeline is located in:

```text
src/pipeline/
```

### Components

| Module | Responsibility |
|---|---|
| `config.py` | Central configuration, paths, schema and model feature definitions |
| `data_ingestion.py` | Loads the raw CSV and checks data structure |
| `data_validation.py` | Validates columns, duplicates, missing values, data types, categories and target distribution |
| `data_transformation.py` | Cleans raw data and creates analytical features |
| `data_pipeline.py` | Main entry point that executes the complete pipeline |

### Pipeline

```text
Raw CSV
   ↓
Ingestion
   ↓
Validation
   ↓
Cleaning
   ↓
Transformation
   ↓
Feature Engineering
   ↓
Processed Dataset
   ↓
Analytics + Machine Learning
```

### Run the pipeline

```bash
python -m src.pipeline.data_pipeline
```

### Pipeline Outputs

The pipeline generates:

```text
data/processed/cleaned_telco_churn.csv
```

Model-ready cleaned dataset.

```text
data/processed/telco_churn_features.csv
```

Cleaned dataset with additional analytics features.

```text
reports/data_quality_report.csv
```

Data quality and validation results.

```text
reports/last_pipeline_run.json
```

Timestamp and summary of the latest pipeline execution.

---

# 🧹 Data Cleaning & Validation

The pipeline performs several data quality checks before the data reaches the analytics and ML stages.

### Validation checks

- Required columns
- Missing values
- Duplicate rows
- Invalid data types
- Unexpected categorical values
- Target distribution
- Dataset shape

### Cleaning operations

- Convert `TotalCharges` to numeric
- Handle blank values
- Remove duplicate records
- Remove unnecessary `customerID`
- Prepare consistent feature types

This prevents invalid or inconsistent data from silently entering downstream processes.

---

# ⚙️ Feature Engineering

The project creates additional features for analytics and customer segmentation.

Examples include:

- `TenureGroup`
- `MonthlyChargeGroup`
- `TotalChargesGroup`
- `ServiceCount`
- `IsLongTermCustomer`
- `IsMonthToMonth`
- `HighValueCustomer`

These features are primarily used by the analytics layer and dashboard.

---

# 📊 Analytics Layer

The analytics layer is located in:

```text
src/analytics/
```

### Components

| Module | Responsibility |
|---|---|
| `churn_analysis.py` | Churn rates and comparisons across customer categories |
| `customer_analysis.py` | Customer filtering, distributions and segmentation |
| `insights.py` | Generates dashboard insights dynamically from the data |

The dashboard does not rely on hardcoded analytical values.

KPIs and charts are calculated from the processed data at runtime.

---

# 🖥️ Interactive Streamlit Dashboard

The project includes a multi-page Streamlit dashboard.

### Dashboard Pages

### 🏠 Overview

Displays:

- Total customers
- Churn rate
- Average charges
- Average tenure
- Revenue
- High-risk customers
- Automatically generated key insights

### 📊 Data Analysis

Provides:

- Dataset statistics
- Missing-value analysis
- Duplicate analysis
- Churn distribution
- Gender distribution
- Contract distribution
- Internet service distribution
- Payment method distribution

### 👥 Customer Analysis

Provides interactive filtering and segmentation based on:

- Contract
- Internet service
- Gender
- Senior citizen status
- Payment method
- Churn status
- Customer tenure
- Customer value
- Churn risk

### 📈 Churn Analysis

Analyzes churn rates across multiple dimensions:

- Contract
- Internet service
- Payment method
- Tenure
- Gender
- Senior citizen status
- Monthly charges
- Total charges
- Number of services

### 💳 Revenue & Charges

Includes:

- Monthly charge distribution
- Average charges by churn status
- Charges vs tenure
- High-value customer analysis

### 🤖 ML Prediction

Allows users to enter customer information and receive:

- Churn prediction
- Churn probability
- Risk level

Risk levels:

```text
Low       → < 30%
Medium    → 30% – 60%
High      → ≥ 60%
```

### ℹ️ Data Quality

Displays:

- Validation results
- Data types
- Column statistics
- Target distribution
- Latest pipeline execution information

---

# 🤖 Machine Learning Pipeline

The machine learning workflow is:

```text
Processed Dataset
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
Feature Encoding
        ↓
Feature Scaling
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Model
        ↓
Saved ML Pipeline
        ↓
Churn Prediction
```

### Models Compared

The project compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Preprocessing

The ML pipeline uses:

- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`

### Hyperparameter Tuning

`GridSearchCV` is used to identify suitable model parameters.

### Final Model

The final tuned Logistic Regression pipeline is saved as:

```text
models/churn_pipeline.joblib
```

The saved pipeline can then be used directly for customer churn prediction.

---

# 🔬 Model Training

The main ML scripts are:

```text
src/model_comparison.py
src/hyperparameter_tuning.py
src/train.py
src/train_final.py
```

### Train / compare models

```bash
python src/model_comparison.py
```

### Run hyperparameter tuning

```bash
python src/hyperparameter_tuning.py
```

### Train the final model

```bash
python src/train_final.py
```

---

# 📁 Project Structure

```text
customer-churn-data-pipeline/
│
├── app/
│   ├── app.py
│   ├── dashboard.py
│   └── prediction.py
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   └── processed/
│       ├── cleaned_telco_churn.csv
│       └── telco_churn_features.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── reports/
│   ├── data_quality_report.csv
│   └── last_pipeline_run.json
│
├── src/
│   ├── pipeline/
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── data_pipeline.py
│   │
│   ├── analytics/
│   │   ├── churn_analysis.py
│   │   ├── customer_analysis.py
│   │   └── insights.py
│   │
│   ├── data_cleaning.py
│   ├── data_inspection.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_comparison.py
│   ├── hyperparameter_tuning.py
│   ├── train.py
│   └── train_final.py
│
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone https://github.com/riddhijosh2006-rgb/customer-churn-data-pipeline.git
```

```bash
cd customer-churn-data-pipeline
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the data pipeline

```bash
python -m src.pipeline.data_pipeline
```

## 5. Launch the dashboard

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# 🧪 Testing & Validation

The project has been tested across the major pipeline components.

### Data Pipeline

The complete data pipeline was executed successfully.

```text
Raw rows:       7,043
Cleaned rows:   7,021
Duplicates removed: 22
```

The pipeline successfully generated the data-quality report and latest pipeline execution log.

### ML Pipeline

The saved machine learning pipeline was tested against the cleaned dataset without column mismatch.

### Dashboard

All dashboard pages were tested, including the ML prediction workflow.

### Code Validation

Python source files were checked using:

```bash
python -m py_compile
```

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| Joblib | Model serialization |
| Git | Version control |
| GitHub | Source control and project documentation |

---

# 📈 Key Engineering Concepts Demonstrated

This project demonstrates practical experience with:

### Data Engineering

- Data ingestion
- Data validation
- Data cleaning
- Data transformation
- Feature engineering
- Data quality reporting
- Pipeline modularization
- Reusable pipeline design

### Data Analytics

- Exploratory Data Analysis
- Customer segmentation
- Churn analysis
- KPI generation
- Business insights
- Interactive filtering
- Data visualization

### Machine Learning

- Binary classification
- Feature preprocessing
- Model comparison
- Logistic Regression
- Random Forest
- Gradient Boosting
- Hyperparameter tuning
- Model serialization
- Prediction probability
- Risk classification

### Deployment

- Streamlit application
- Reusable ML pipeline
- Interactive prediction interface

---

# 💡 Business Value

The project can help a telecom business answer questions such as:

- Which customers are most likely to churn?
- Which contracts have the highest churn rate?
- How does tenure affect customer retention?
- Which services are associated with higher churn?
- Which customers have high churn risk?
- Which customer segments should receive retention efforts?

The ML prediction layer adds a forward-looking component by identifying customers who may be at higher risk of leaving.

---

# 🚧 Future Improvements

Planned improvements include:

- [ ] Apache Airflow orchestration
- [ ] Automated scheduled pipeline execution
- [ ] Automated data-quality testing
- [ ] Unit and integration tests
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Model monitoring
- [ ] Model drift detection
- [ ] Cloud deployment
- [ ] Automated model retraining
- [ ] Production API for model inference

---

# 👨‍💻 About

Built as a practical project to explore the intersection of:

**Data Engineering → Data Analytics → Machine Learning → Deployment**

The project focuses on building a reusable workflow rather than creating a standalone machine learning notebook.

---

## ⭐ Project Highlights

```text
✅ Reusable Data Pipeline
✅ Data Validation
✅ Data Quality Reporting
✅ Feature Engineering
✅ Customer Analytics
✅ Multiple ML Models
✅ Hyperparameter Tuning
✅ Saved ML Pipeline
✅ Churn Probability
✅ Customer Risk Levels
✅ Interactive Streamlit Dashboard
```

---

## 📫 Connect

**GitHub:**  
https://github.com/riddhijosh2006-rgb
