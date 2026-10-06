# Detecting Credit Card Fraud with Machine Learning

UE24CS352A – Machine Learning · Mini-Project

**Team:** PES2UG24AM066 · PES2UG24AM014

Compares classical ML models and class-imbalance techniques for detecting fraudulent
credit card transactions. Based on Rosenbaum, *Detecting Credit Card Fraud with Machine
Learning*, Stanford CS229 (2019), re-implemented in Python with extensions
(SMOTE, class weighting, XGBoost, threshold tuning, demo app).

## Problem
Fraud is ~0.17% of transactions, so accuracy is meaningless (predicting "legit" always gives
99.8%). We use **AUPRC** (area under the precision-recall curve) as the primary metric and
compare models across several imbalance-handling strategies.

## Dataset
[Kaggle – Credit Card Fraud Detection (ULB)](https://www.kaggle.com/mlg-ulb/creditcardfraud):
284,807 transactions, 492 frauds. Columns: `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class`.

The CSV is **not** committed (≈150 MB). Download `creditcard.csv` and place it at `data/creditcard.csv`.

## Setup
```bash
git clone <your-repo-url> && cd credit-card-fraud-detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ recommended.

## Run
```bash
# 1. Full experiment (5 models x 5 strategies, validation + test evaluation)
python -m src.train --data data/creditcard.csv

# Quick smoke test (subsampled, smaller models, a few minutes)
python -m src.train --data data/creditcard.csv --fast

# Subset of models / strategies
python -m src.train --models rf xgboost --strategies none smote

# 2. Effect of resampling ratio p (paper's Figure 2)
python -m src.sweep_p --data data/creditcard.csv

# 3. Demo app (needs results/model.joblib from step 1)
streamlit run app.py
```
Outputs land in `results/`: `validation_results.csv`, `test_results.csv`,
`pr_curves_test.png`, `confusion_*.png`, `sweep_p.png`, `model.joblib`.

No dataset handy? `python scripts/make_synthetic_data.py data/synthetic.csv` creates a fake file
with the same schema for debugging only (do not report its numbers).

## Method
1. **Split:** stratified 2/3 train, 1/6 validation, 1/6 test (seeded).
2. **Preprocessing:** standardise `Time` and `log(1+Amount)`; scaler fit on train only.
3. **Imbalance strategies** (train split only): none · class weights · random oversampling ·
   random undersampling · SMOTE. Resampled minority fraction `p = 0.02` by default.
4. **Models:** logistic regression · quadratic logistic regression + L1 (LASSO) · random forest ·
   XGBoost · MLP.
5. **Selection:** best strategy per model by validation AUPRC; F1-optimal threshold chosen on
   validation; finalists evaluated once on the test set.

## Repository layout
```
src/data.py        loading, stratified split, Preprocessor
src/sampling.py    over / under / SMOTE samplers
src/models.py      model definitions
src/evaluate.py    metrics, threshold selection, plots
src/train.py       full experiment
src/sweep_p.py     resampling-ratio sweep
app.py             Streamlit demo
scripts/           synthetic data helper
notebooks/         EDA
results/           generated outputs
report/            write-up PDF
```

## Results
_Fill in after running (copy from `results/test_results.csv`)._

| Model | Strategy | AUPRC | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| | | | | | | |

## Contributions
| SRN | Contribution |
|---|---|
| PES2UG24AM066 | |
| PES2UG24AM014 | |

## Reference
A. Rosenbaum, "Detecting Credit Card Fraud with Machine Learning," CS229, Stanford University, 2019.
