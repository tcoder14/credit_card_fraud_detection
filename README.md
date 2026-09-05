# Credit Card Fraud Detection Using Machine Learning Classification Algorithms

A machine learning project that builds and compares six supervised classifiers for detecting fraudulent credit card transactions, using the **Credit Card Fraud Detection Dataset 2023** (568,630 anonymized European cardholder transactions).

This repository was developed as a group academic project.

## Overview

Credit card companies process huge volumes of transactions every day, making manual fraud review impractical. This project builds a reproducible pipeline that:

- Cleans and preprocesses an anonymized transaction dataset
- Trains six classical and ensemble machine learning models
- Evaluates every model with the same metrics and the same train-test split, for a fair, apples-to-apples comparison
- Analyzes which models are best suited to this kind of fraud classification task

## Repository Contents

| File | Description |
|---|---|
| `credit_card_fraud_detection.ipynb` | Jupyter notebook containing all data preprocessing, exploratory analysis, model training, and evaluation code |
| `CreditCardFraud_IEEE_Report.pdf` | Final project report, written in IEEE conference paper format |
| `README.md` | This file |

## Dataset

- **Name:** Credit Card Fraud Detection Dataset 2023
- **Source:** [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **Size:** 568,630 transactions, 31 columns (a transaction `id`, anonymized features `V1`–`V28`, `Amount`, and binary target `Class`)
- **Class balance:** Exactly balanced — 284,315 legitimate and 284,315 fraudulent transactions (unlike most real-world fraud datasets, which are heavily imbalanced)

## Methodology

1. **Preprocessing**
   - Removed the `id` column (no predictive signal)
   - Removed one duplicate transaction (568,630 → 568,629 rows)
   - No missing values and no categorical encoding needed
   - Standardized the `Amount` column with `StandardScaler` (fit on training data only)
   - Stratified 80:20 train-test split (`random_state=42`)

2. **Models trained**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
   - SGD-based Support Vector Machine

3. **Evaluation metrics**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9648 | 0.9771 | 0.9519 | 0.9644 | 0.9935 |
| Decision Tree | 0.9956 | 0.9944 | 0.9968 | 0.9956 | 0.9972 |
| Random Forest | 0.9992 | 0.9993 | 0.9991 | 0.9992 | 1.0000 |
| **XGBoost** | **0.9997** | 0.9995 | **1.0000** | **0.9997** | 1.0000 |
| LightGBM | 0.9993 | 0.9985 | **1.0000** | 0.9993 | 0.9999 |
| SVM (SGD) | 0.9648 | 0.9763 | 0.9528 | 0.9644 | 0.9932 |

**XGBoost** delivered the strongest overall performance, correctly identifying all fraud cases in the held-out test set with only 31 false positives out of 56,863 legitimate transactions. Ensemble and boosting tree methods (Random Forest, XGBoost, LightGBM) consistently outperformed the two linear baselines (Logistic Regression, SGD-based SVM), particularly on recall — the metric that matters most for catching fraud.

Full details, discussion, limitations, and comparison with related literature are in the [report](./CreditCardFraud_IEEE_Report.pdf).

## Tools & Libraries

- Python 3
- pandas, NumPy — data loading and manipulation
- scikit-learn — Logistic Regression, Decision Tree, Random Forest, SGDClassifier, preprocessing, evaluation metrics
- xgboost, lightgbm — gradient boosting models
- Matplotlib — visualizations
- Jupyter Notebook

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib jupyter

# Launch the notebook
jupyter notebook credit_card_fraud_detection.ipynb
```

The dataset itself is not included in this repository due to size; download it from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) and place it in the working directory before running the notebook.

## Limitations & Future Work

- The dataset is artificially balanced (50:50); real-world fraud rates are typically under 1%, so results may not directly generalize to production traffic.
- The anonymized `V1`–`V28` features limit interpretability of *why* a transaction is flagged.
- Results are based on a single stratified train-test split with default-style hyperparameters rather than full cross-validation and tuning.

Planned extensions include testing under realistic class imbalance, systematic hyperparameter tuning, SHAP-based explainability, and evaluating generalization to a different fraud dataset. See the report's Conclusion section for details.
