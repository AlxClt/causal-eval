"""
Demo: train a propensity model and evaluate it with causaleval.

Data: NHEFS (Hernan & Robins, "Causal Inference: What If"), loaded through the
`causaldata` package (demo-only dependency: `pip install causaldata`).
    - treatment: qsmk, quitting smoking between 1971 and 1982
    - outcome:   wt82_71, weight change in kg
    - covariates: sex, race, age, education, smoking intensity and years,
                  exercise, activity, baseline weight

Steps:
    1. Load the NHEFS data
    2. Train propensity models (logistic regression, random forest, calibrated random forest) with 5-fold
       cross-validation
    3. Evaluate each model (covariate balance, calibration, propensity distribution, ROC curves)
       and save the plots in demo/plots/
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import causaleval
from utils import summarize_fold_metrics

TREATMENT = "qsmk"
OUTCOME = "wt82_71"
CATEGORICALS = ["sex", "race", "education", "exercise", "active"]
CONTINUOUS = ["age", "smokeintensity", "smokeyrs", "wt71"]
COVARIATES = CATEGORICALS + CONTINUOUS

# causaleval's evaluation functions expect the treatment column to be named 'treated'
TREATED_COL = "treated"

MODELS = ["logistic", "random_forest", "calibrated_random_forest"]

PLOTS_DIR = Path(__file__).parent / "plots"


def load_data():
    try:
        from causaldata import nhefs_complete
    except ImportError as e:
        raise ImportError("The demo needs the causaldata package: pip install causaldata") from e

    df = nhefs_complete.load_pandas().data
    df = df[COVARIATES + [TREATMENT, OUTCOME]].copy()
    df[TREATED_COL] = df[TREATMENT].astype(int)
    return df.reset_index(drop=True)


def build_pipeline(model="logistic", seed=42):
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICALS),
        ("continuous", StandardScaler(), CONTINUOUS),
    ])

    # a minimum leaf size avoids overconfident (close to 0 or 1) propensities
    random_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, n_jobs=-1, random_state=seed)

    if model == "logistic":
        classifier = LogisticRegression(max_iter=1000)
    elif model == "random_forest":
        classifier = random_forest
    elif model == "calibrated_random_forest":
        # sigmoid (Platt) calibration fitted with an inner 5-fold CV on the training fold only;
        # isotonic calibration would overfit at this sample size and can output propensities of exactly 0 or 1
        classifier = CalibratedClassifierCV(random_forest, method="sigmoid", cv=5)
    else:
        raise ValueError(f"Unknown model '{model}', use one of {MODELS}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


@dataclass
class PropensityResult:
    """
    data: input data + out-of-fold 'propensity', 'ip_weight' and 'fold' columns
    fold_metrics: one row per fold, one column per metric
    models: fitted pipeline of each fold
    """
    model: str
    data: pd.DataFrame
    fold_metrics: pd.DataFrame
    models: list

    def fold_predictions(self):
        """(y_true, y_score) lists with one array per fold"""
        folds = [self.data[self.data["fold"] == k] for k in sorted(self.data["fold"].unique())]
        return [f[TREATED_COL].values for f in folds], [f["propensity"].values for f in folds]


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }


def train_propensity_model(data, model="logistic", n_splits=5, seed=42, clip=(0.01, 0.99)):
    X = data[COVARIATES]
    y = data[TREATED_COL].values

    propensity = np.zeros(len(data))
    fold = np.zeros(len(data), dtype=int)
    metrics, models = [], []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        pipeline = build_pipeline(model, seed=seed)
        pipeline.fit(X.iloc[train_idx], y[train_idx])

        y_score = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        propensity[test_idx] = y_score
        fold[test_idx] = k
        metrics.append(compute_metrics(y[test_idx], y_score))
        models.append(pipeline)

    scored = data.copy()
    scored["propensity"] = propensity
    p = np.clip(propensity, *clip)
    scored["ip_weight"] = y / p + (1 - y) / (1 - p)
    scored["fold"] = fold

    fold_metrics = pd.DataFrame(metrics).rename_axis("fold")
    return PropensityResult(model=model, data=scored, fold_metrics=fold_metrics, models=models)


def write_figures_html(figs, path):
    """Write several plotly figures, one below the other, in a single HTML file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    divs = [fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, fig in enumerate(figs)]
    path.write_text(f"<html><head><meta charset='utf-8'></head><body>{''.join(divs)}</body></html>", encoding="utf-8")
    return path


def plot_confounder_distributions(data, out_dir=PLOTS_DIR):
    """Distribution of each covariate in the control and treated groups (model independent)"""
    figs = causaleval.plot_confounder_distributions(data, COVARIATES, treatment=TREATED_COL)
    for covariate, fig in zip(COVARIATES, figs):
        fig.update_layout(title=f"Distribution of {covariate}", showlegend=False)
    return write_figures_html(figs, Path(out_dir) / "confounder_distributions.html")


def evaluate_model(result, out_dir=PLOTS_DIR):
    """Save the propensity evaluation plot and the confounding evidence plots of a model, return their paths"""
    fig = causaleval.plot_propensity_evaluation(
        result.data, CONTINUOUS, CATEGORICALS, fold="fold",
        metrics_summary=summarize_fold_metrics(result.fold_metrics),
        title=f"Propensity model evaluation: {result.model}")
    evaluation_path = write_figures_html([fig], Path(out_dir) / f"propensity_evaluation_{result.model}.html")

    evidence_figs = []
    for covariate in COVARIATES:
        fig = causaleval.plot_confounding_evidence(result.data, "propensity", covariate,
                                                   categorical=covariate in CATEGORICALS)
        fig.update_layout(title=f"Propensity by {covariate}: {result.model}")
        evidence_figs.append(fig)
    evidence_path = write_figures_html(evidence_figs, Path(out_dir) / f"confounding_evidence_{result.model}.html")

    return evaluation_path, evidence_path


def main():
    print(f"causaleval {causaleval.__version__}")
    data = load_data()
    print(f"NHEFS: {len(data)} rows, {data[TREATED_COL].mean():.1%} treated ({TREATMENT})")

    print(f"confounder distributions saved to {plot_confounder_distributions(data)}")

    results = {model: train_propensity_model(data, model=model) for model in MODELS}

    for model, result in results.items():
        print(f"\n=== {model} ===")
        print(result.fold_metrics.round(3).to_string())
        print("\nout-of-sample metrics (95% CI over folds):")
        print(summarize_fold_metrics(result.fold_metrics).round(3).to_string())
        for path in evaluate_model(result):
            print(f"plot saved to {path}")

    return results


if __name__ == "__main__":
    main()
