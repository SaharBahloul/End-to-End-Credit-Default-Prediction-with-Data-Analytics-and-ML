# End-to-End-Credit-Default-Prediction-with-Data-Analytics-and-ML
An end-to-end credit-risk prediction system using the Kaggle Home Credit dataset. Includes data analytics, preprocessing, SMOTE imbalance handling, multiple ML models (RF, XGBoost, LightGBM, CatBoost), threshold optimization, and deployment of the final CatBoost model in a Streamlit app for real-time scoring.

This project presents an end-to-end credit-risk prediction system combining advanced data analytics, machine-learning modeling, and practical deployment. Using the Home Credit Default Risk dataset from Kaggle, the pipeline begins with a comprehensive exploratory analysis addressing missingness patterns, feature correlations, sentinel values, and nonlinearity. A robust preprocessing framework is implemented, including outlier clipping, winsorization, semantic corrections, engineered ratios (e.g., CREDIT/INCOME, ANNUITY/INCOME), PCA exploration, rare-category consolidation, and scaling. Given the severe class imbalance typical in credit-scoring problems, multiple imbalance-handling strategies are evaluated, including SMOTE oversampling and threshold engineering.

A full suite of models—Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost—is trained both before and after SMOTE, and assessed using metrics appropriate for imbalanced data (ROC-AUC, PR-AUC, recall, precision, and F1-score). Additional experiments explore precision-driven thresholds, Youden index, and class-wise splitting to understand model behavior under different decision constraints. Gradient-boosting methods consistently outperform classical models, with CatBoost delivering the strongest minority-class performance. After correcting a detected test-leakage issue and retraining the model on a strictly validated pipeline, CatBoost is selected as the final model.

To bridge experimentation with real-world decision-making, the model and preprocessing artifacts are deployed in a Streamlit application. The app enables batch CSV prediction, manual feature input, adjustable decision thresholds, and real-time probability scoring, making the solution accessible for analysts and non-technical stakeholders. The resulting system demonstrates how modern data science techniques can be integrated to build accurate, interpretable, and operational credit-risk modeling tools.

Full credit-risk prediction pipeline that includes:

Data analytics

Exploratory data analysis (EDA)

Missing-value analysis

Correlation studies (Pearson, Spearman, Cramér’s V)

Feature distribution inspection

Class imbalance diagnostics

Outlier analysis

Data preprocessing & feature engineering

Removing constant/high-missing features

Fixing sentinel values (e.g., DAYS_EMPLOYED = 365243)

Winsorizing high-variance numeric fields

Ratio-engineering (CREDIT/INCOME, ANNUITY/INCOME, AGE_YEARS, etc.)

Rare-category collapsing

Scaling

PCA analysis

Imbalanced classification experimentation

Baseline models (RF, Logistic)

Threshold optimization (0.5, Youden, best-F1, precision≥0.80)

Alternative stratified splitting

SMOTE oversampling

Retraining models before/after resampling

Advanced machine-learning modeling

Random Forest

Logistic Regression

XGBoost

LightGBM

CatBoost (best model)

Comparing models using ROC-AUC, PR-AUC, precision, recall, F1

Ensemble of multiple boosting models

Model tuning and validation

Hyperparameter tuning (CatBoost RS-CV)

Anti-leakage retraining

Threshold calibration

Final selection of CatBoost

Deployment and productionization

Packaging model artifacts (model, scaler, feature list)

Building a Streamlit app for inference

Supporting CSV batch upload + manual input

Threshold-controlled predictions

Reproducible, usable credit-risk scoring tool
