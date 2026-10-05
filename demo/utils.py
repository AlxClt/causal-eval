import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def summarize_fold_metrics(fold_metrics, level=0.95, bounds=(0, 1)):
    """
    fold_metrics: pd.DataFrame with one row per cross-validation fold and one column per metric
    Returns the mean, standard deviation and Student t confidence interval of each metric over the folds,
    in the format expected by causaleval.plot_metrics_summary / plot_propensity_evaluation.
    The interval is approximate: folds share training data, so it tends to be too narrow.
    bounds: range of the metrics the interval is clipped to (None to disable)
    """
    n = len(fold_metrics)
    mean, std = fold_metrics.mean(), fold_metrics.std(ddof=1)
    half_width = student_t.ppf((1 + level) / 2, df=n - 1) * std / np.sqrt(n)
    low, high = mean - half_width, mean + half_width
    if bounds is not None:
        low, high = low.clip(*bounds), high.clip(*bounds)
    return pd.DataFrame({'mean': mean, 'std': std, 'ci_low': low, 'ci_high': high})
