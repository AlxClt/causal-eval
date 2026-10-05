# Demo

`propensity_demo.py` trains propensity models on the NHEFS observational dataset and evaluates them with `causaleval`.

## Data

NHEFS (National Health and Nutrition Examination Survey Epidemiologic Follow-up Study), as used in Hernán & Robins, *Causal Inference: What If*:

- treatment: `qsmk`, quitting smoking
- outcome: `wt82_71`, weight change in kg
- covariates: sex, race, age, education, smoking intensity and years, exercise, activity, baseline weight

The data is downloaded with the [`causaldata`](https://pypi.org/project/causaldata/) package, which must be manually installed as it is not a dependency of `causaleval`

## Training

Three models (scikit-learn pipelines: one-hot encoding of categoricals, scaling of continuous covariates) are trained with 5-fold stratified cross-validation:

- `logistic`: logistic regression
- `random_forest`: random forest
- `calibrated_random_forest`: random forest wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)`. The calibration is fitted with an inner cross-validation on each training fold only, so the evaluation stays out-of-sample.

Note that for the calibraion, sigmoid (Platt) is preferred to isotonic at this sample size: isotonic tends to overfit on small datasets

For each fold, ROC AUC, accuracy, F1, precision, recall and Brier score are recorded on the held-out data as informative metrics.

`train_propensity_model` returns a `PropensityResult` with:

- `data`: the dataset with a `treated` column plus out-of-fold `propensity`, `ip_weight` and `fold` columns
- `fold_metrics`: the metrics of each fold
- `models`: the fitted pipeline of each fold
- `fold_predictions()`: per-fold true treatment / propensity arrays

## Evaluation

`evaluate_model` calls `causaleval.plot_propensity_evaluation` (modelled on Fig. 3 of the [paper](https://arxiv.org/pdf/1906.00442)) and saves one interactive plot per model in `demo/plots/propensity_evaluation_<model>.html`. Its panels:

- **A. Covariate balance:** absolute standardized mean difference of each covariate before (original) and after (weighted) inverse propensity weighting. Weighted values should be below 0.1.
- **B. Calibration:** observed treatment rate vs predicted propensity, one curve per cross-validation fold. Curves should follow the diagonal.
- **C. Propensity distribution:** propensities of treated and control. Overlap between the two is needed for the weights to be stable.
- **D. ROC curves:** the propensity model's ROC curve should be close to the *expected* curve (what the model's own propensities imply), and the *IP-weighted* curve should be close to the diagonal (the weighted data looks randomized).
- **E. Out-of-sample metrics:** mean, 95% confidence interval and standard deviation of each metric over the 5 folds. The interval is a Student t interval (4 degrees of freedom) clipped to [0, 1]. It is approximate and tends to be too narrow, because folds share training data.

Each panel is also available on its own: `plot_standardized_differences`, `plot_calibration_curves`, `plot_propensity_distribution`, `plot_roc_curves`, `plot_metrics_summary`. The table values are computed by `summarize_fold_metrics` in [utils.py](utils.py), which is part of the demo, not of the package.

### Confounders

Also saved in `demo/plots/`:

- `confounder_distributions.html`: distribution of each covariate in the control and treated groups (`plot_confounder_distributions`).
- `confounding_evidence_<model>.html`: predicted propensity by category, or by decile for continuous covariates (`plot_confounding_evidence`). A trend across categories shows that the covariate drives treatment, so that it is a candidate confounder.

## Run

From the repository root:

```bash
pip install -e .
pip install causaldata
python demo/propensity_demo.py
```
