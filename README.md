# End-to-End-Credit-Default-Prediction-with-Data-Analytics-and-ML
An end-to-end credit-risk prediction system using the Kaggle Home Credit dataset. Includes data analytics, preprocessing, SMOTE imbalance handling, multiple ML models (RF, XGBoost, LightGBM, CatBoost), threshold optimization, and deployment of the final CatBoost model in a Streamlit app for real-time scoring.

This project presents an end-to-end credit-risk prediction system combining advanced data analytics, machine-learning modeling, and practical deployment. Using the Home Credit Default Risk dataset from Kaggle, the pipeline begins with a comprehensive exploratory analysis addressing missingness patterns, feature correlations, sentinel values, and nonlinearity. A robust preprocessing framework is implemented, including outlier clipping, winsorization, semantic corrections, engineered ratios (e.g., CREDIT/INCOME, ANNUITY/INCOME), PCA exploration, rare-category consolidation, and scaling. Given the severe class imbalance typical in credit-scoring problems, multiple imbalance-handling strategies are evaluated, including SMOTE oversampling and threshold engineering.

A full suite of models—Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost—is trained both before and after SMOTE, and assessed using metrics appropriate for imbalanced data (ROC-AUC, PR-AUC, recall, precision, and F1-score). Additional experiments explore precision-driven thresholds, Youden index, and class-wise splitting to understand model behavior under different decision constraints. Gradient-boosting methods consistently outperform classical models, with CatBoost delivering the strongest minority-class performance. After correcting a detected test-leakage issue and retraining the model on a strictly validated pipeline, CatBoost is selected as the final model.

To bridge experimentation with real-world decision-making, the model and preprocessing artifacts are deployed in a Streamlit application. The app enables batch CSV prediction, manual feature input, adjustable decision thresholds, and real-time probability scoring, making the solution accessible for analysts and non-technical stakeholders. The resulting system demonstrates how modern data science techniques can be integrated to build accurate, interpretable, and operational credit-risk modeling tools.

📊 1. Data Analytics

Exploratory Data Analysis (EDA)

Missing-value analysis

Correlation studies (Pearson, Spearman, Cramér’s V)

Feature distribution inspection

Class imbalance diagnostics

Outlier analysis

🧹 2. Data Preprocessing & Feature Engineering

Removal of constant and high-missingness features

Sentinel-value corrections (e.g., DAYS_EMPLOYED = 365243)

Winsorization of extreme numeric values

Engineered ratios (CREDIT/INCOME, ANNUITY/INCOME, AGE_YEARS, etc.)

Rare-category collapsing for sparse categorical variables

Scaling of numeric features

PCA analysis for variance structure

⚖️ 3. Imbalanced Classification Experiments

Baseline models (Random Forest, Logistic Regression)

Threshold optimization:

0.50 default threshold

Youden index

Best-F1 threshold

Precision ≥ 80% thresholding

Alternative stratified train–test splitting

SMOTE oversampling

Retraining models pre- and post-SMOTE

🤖 4. Advanced Machine-Learning Modeling

Random Forest

Logistic Regression

XGBoost

LightGBM

CatBoost (best model)

Model comparison via:

ROC-AUC

PR-AUC

Precision

Recall

F1-score

Ensemble of boosting models

🔧 5. Model Tuning and Validation

CatBoost randomized search (RS-CV)

Anti-leakage retraining

Threshold calibration for risk-sensitive decisions

Final selection of the optimized CatBoost classifier

📦 6. Deployment & Productionization

Exporting all artifacts (CatBoost model, scaler, feature list)

Building an interactive Streamlit app for inference

Supports:

CSV batch predictions

Manual feature input

Adjustable decision thresholds

Fully reproducible and ready-to-use credit-risk scoring tool
