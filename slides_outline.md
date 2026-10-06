# Slide deck outline (target 10–12 slides, ~10 min + live demo)

1. **Title** – project name, course, SRNs, date.
2. **Problem & motivation** – fraud cost, why ML, why it is hard (imbalance). One striking number: 0.173% fraud.
3. **Reference paper** – Rosenbaum CS229: models compared, best result (RF + oversampling, AUPRC ≈ 0.91), what we reproduce vs extend.
4. **Dataset** – 284,807 rows, 492 frauds, feature types; class-balance chart; PCA pair-plot.
5. **Why accuracy fails / metrics** – naive classifier = 99.8% accuracy; define precision, recall, PR curve, AUPRC.
6. **Pipeline diagram** – split → preprocess → resample (train only) → train → validate → threshold → test. Call out leakage prevention.
7. **Imbalance strategies** – none, class weight, over, under, SMOTE (one line each) + the p-sweep figure.
8. **Models** – logistic, quadratic + LASSO, RF, XGBoost, MLP (one line each, key hyper-parameters).
9. **Validation results** – heat-map of model × strategy AUPRC.
10. **Test results** – results table + PR curves + winner's confusion matrix; threshold-tuning effect.
11. **Comparison with paper & insights** – what matched, what differed, why.
12. **Live demo** – Streamlit app: score sample, show flagged rows and probabilities. (Have a backup screenshot.)
13. **Conclusions, limitations, future work; contributions per member.**

## Q&A prep (both members should answer any of these)
- Why AUPRC over ROC-AUC or accuracy? (ROC-AUC is inflated by the huge negative class.)
- Why resample only the training set? (Otherwise duplicated/synthetic rows leak into evaluation.)
- How does SMOTE work, and its downsides? (Interpolates between minority neighbours; can add noise.)
- Why did random forest / XGBoost beat logistic regression (or not)?
- How was the threshold chosen, and why on validation not test?
- What does L1 do in the quadratic model? (Zeroes coefficients; combats ~500-feature overfit.)
- What would you change for production? (Cost-based threshold, drift monitoring, online updates.)
- Who wrote which module? Walk through `train.py`.
