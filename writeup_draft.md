# Detecting Credit Card Fraud with Machine Learning
**UE24CS352A – ML Mini-Project** · PES2UG24AM066, PES2UG24AM014 · GitHub: <repo link>

*(Layout note: your guidelines say both "one-page" and "two-page". Confirm with faculty. This draft targets 2 pages; trim the italic optional lines for 1 page.)*

## 1. Problem Statement
Payment fraud costs billions annually, and banks rely on ML to flag suspicious transactions.
The central difficulty is extreme class imbalance: only 0.173% of transactions are fraudulent,
so a model that always predicts "legitimate" is 99.8% accurate yet useless. Our goal is to build
and compare classifiers that detect fraud with high precision and recall, and to study which
imbalance-handling technique works best. We follow the setup of Rosenbaum (CS229, 2019).

## 2. Dataset
Kaggle / ULB Credit Card Fraud dataset: 284,807 European card transactions over two days,
492 fraudulent. Features: `Time`, `Amount`, and 28 anonymised PCA components `V1`–`V28`
(original features withheld for privacy); label `Class` (1 = fraud). Because features are
anonymised, we focus on predictive performance, not interpretation. We use a stratified
2/3–1/6–1/6 train/validation/test split so each split keeps the fraud ratio
(train [__] / val [__] / test [__] frauds).

## 3. Approach
- **Preprocessing:** standardise `Time` and `log(1+Amount)` (scaler fit on train only).
- **Imbalance strategies (train split only):** none, class weights, random oversampling,
  random undersampling, SMOTE; minority fraction p = [__] (chosen by validation sweep).
- **Models:** logistic regression; quadratic logistic regression with L1 (LASSO);
  random forest; XGBoost; MLP.
- **Metric:** AUPRC (primary), plus AUROC, precision, recall, F1 and confusion matrices.
  Accuracy is not used for model selection.
- **Protocol:** every (model, strategy) pair is trained on the training split and scored on
  validation; the best strategy per model is kept; the decision threshold is tuned on
  validation to maximise F1; finalists are evaluated once on the test set.

## 4. Implementation Overview
Python (scikit-learn, imbalanced-learn, XGBoost). Modular code: `data.py` (split and
preprocessing), `sampling.py`, `models.py`, `evaluate.py`, `train.py` (experiment driver),
`sweep_p.py` (resampling-ratio study) and a Streamlit app (`app.py`) that scores uploaded or
sampled transactions with the best saved model. Resampling is applied only after the split to
prevent leakage. All runs are seeded; setup steps are in the README.

**Table 1 – Test-set results** *(copy from results/test_results.csv)*

| Model (strategy) | AUPRC | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| [ ] | | | | | |

*Figure 1: PR curves of finalists (results/pr_curves_test.png). Figure 2: AUPRC vs p (results/sweep_p.png).*

## 5. Conclusions
*(Rewrite once you have numbers. Prompts below.)*
- Best model: [__] with AUPRC [__], recall [__] at precision [__].
- Tree ensembles [did / did not] outperform linear models; the paper found random forest best (AUPRC ≈ 0.91).
- Effect of resampling: [helped which models? SMOTE vs random oversampling vs class weights?]
- Threshold tuning changed F1 from [__] to [__].
- Compared with the paper: [agree / differ, and why, e.g. different library, split seed].
- Limitations: PCA features prevent interpretation; two days of data; static model while fraud
  patterns drift; threshold should reflect real cost of a missed fraud vs a false alarm.
- Future work: cost-sensitive thresholding, streaming/online learning, stacking.

**Reference:** A. Rosenbaum, "Detecting Credit Card Fraud with Machine Learning," Stanford CS229, 2019.
