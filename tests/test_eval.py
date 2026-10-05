import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.metrics import roc_auc_score

import causaleval
from causaleval.eval import standardized_difference

CONTINUOUS = ["age"]
CATEGORICALS = ["sex", "education"]


def two_groups(treated_values, control_values, column="x"):
    return pd.DataFrame({
        column: list(treated_values) + list(control_values),
        "treated": [1] * len(treated_values) + [0] * len(control_values),
    })


# ---------------------------------------------------------------------------
# Standardized differences
# ---------------------------------------------------------------------------

def test_continuous_standardized_difference_known_value():
    df = two_groups([1, 2, 3], [2, 3, 4])
    assert standardized_difference(df, "x", "treated", mode="continuous") == pytest.approx(1)


def test_standardized_difference_keeps_sign():
    df = two_groups([1, 2, 3], [2, 3, 4])
    assert standardized_difference(df, "x", "treated", mode="continuous", return_absolute=False) == pytest.approx(-1)


def test_binary_categorical_standardized_difference_matches_formula():
    df = two_groups([1, 1, 0, 0], [1, 0, 0, 0])
    p_t, p_c = 0.5, 0.25
    expected = abs(p_t - p_c) / np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    assert standardized_difference(df, "x", "treated", mode="categorical") == pytest.approx(expected)


@pytest.mark.parametrize("mode, values", [("continuous", [1, 2, 5, 7]), ("categorical", ["a", "b", "b", "c"])])
def test_identical_groups_have_no_standardized_difference(mode, values):
    df = two_groups(values, values)
    assert standardized_difference(df, "x", "treated", mode=mode) == pytest.approx(0)


def test_fill_imputes_continuous_values_with_group_means():
    with_nan = two_groups([1, 2, np.nan, 3], [2, np.nan, 4])
    filled_by_hand = two_groups([1, 2, 2, 3], [2, 3, 4])
    with pytest.warns(UserWarning, match="Imputing"):
        d = standardized_difference(with_nan, "x", "treated", mode="continuous", handle_nan="fill")
    assert np.isfinite(d)
    assert d == pytest.approx(standardized_difference(filled_by_hand, "x", "treated", mode="continuous"))


@pytest.mark.parametrize("covariate, mode", [("age", "continuous"), ("sex", "categorical")])
def test_true_ip_weights_remove_imbalance(synthetic_data, covariate, mode):
    unweighted = standardized_difference(synthetic_data, covariate, "treated", mode=mode)
    weighted = standardized_difference(synthetic_data, covariate, "treated", weights="ip_weight", mode=mode)
    assert weighted < 0.1
    assert weighted < unweighted


def test_compute_standardized_differences_table(synthetic_data):
    data = synthetic_data.rename(columns={"treated": "T"})
    table = causaleval.compute_standardized_differences(
        [data, data], CONTINUOUS, CATEGORICALS, ["original", "weighted"], weights=[None, "ip_weight"], treatment="T")
    assert list(table.index) == CONTINUOUS + CATEGORICALS
    assert list(table.columns) == ["original", "weighted"]
    assert table.notna().all().all()


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def test_propensity_models_roc_curves_structure():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.4, 0.5, 0.1])
    curves = causaleval.get_propensity_models_roc_curves(y, p)

    assert set(curves) == {"vanilla", "IP_weighted", "Expected"}
    np.testing.assert_allclose(curves["IP_weighted"][2], [1 / 0.8, 1 / 0.6, 1 / 0.5, 1 / 0.9])
    assert all(len(array) == 2 * len(y) for array in curves["Expected"])


def test_roc_curves_with_true_propensities(synthetic_data):
    curves = causaleval.get_propensity_models_roc_curves(synthetic_data["treated"].values,
                                                          synthetic_data["propensity"].values)
    auc = {name: roc_auc_score(y, score, sample_weight=w) for name, (y, score, w) in curves.items()}
    assert auc["IP_weighted"] == pytest.approx(0.5, abs=0.05)
    assert auc["vanilla"] == pytest.approx(auc["Expected"], abs=0.05)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plot", [
    lambda df: causaleval.plot_propensity_evaluation(df, CONTINUOUS, CATEGORICALS, fold="fold"),
    lambda df: causaleval.plot_standardized_differences(df, CONTINUOUS, CATEGORICALS, highlight=["age"]),
    lambda df: causaleval.plot_calibration_curves(df),
    lambda df: causaleval.plot_propensity_distribution(df),
    lambda df: causaleval.plot_roc_curves(df),
    lambda df: causaleval.plot_confounding_evidence(df, "propensity", "education"),
    lambda df: causaleval.plot_confounding_evidence(df, "propensity", "age", categorical=False),
], ids=["evaluation", "standardized_differences", "calibration", "distribution", "roc",
        "evidence_categorical", "evidence_continuous"])
def test_plots_return_figures(synthetic_data, plot):
    assert isinstance(plot(synthetic_data), go.Figure)


def test_plot_metrics_summary_returns_figure(metrics_summary):
    assert isinstance(causaleval.plot_metrics_summary(metrics_summary), go.Figure)


def test_plot_confounder_distributions_returns_one_figure_per_confounder(synthetic_data):
    figs = causaleval.plot_confounder_distributions(synthetic_data, ["age", "education"])
    assert len(figs) == 2
    assert all(isinstance(fig, go.Figure) for fig in figs)


def test_propensity_evaluation_metrics_table_and_title(synthetic_data, metrics_summary):
    def n_tables(fig):
        return sum(isinstance(trace, go.Table) for trace in fig.data)

    assert n_tables(causaleval.plot_propensity_evaluation(synthetic_data, CONTINUOUS, CATEGORICALS)) == 0

    fig = causaleval.plot_propensity_evaluation(synthetic_data, CONTINUOUS, CATEGORICALS,
                                                metrics_summary=metrics_summary, title="My model")
    assert n_tables(fig) == 1
    assert fig.layout.title.text == "<b>My model</b>"
    assert fig.layout.title.x == 0.5


def test_standardized_differences_without_weights(synthetic_data):
    fig = causaleval.plot_standardized_differences(synthetic_data, CONTINUOUS, CATEGORICALS, weights=None)
    assert [trace.name for trace in fig.data] == ["original"]


def test_calibration_curves_one_per_fold(synthetic_data):
    fig = causaleval.plot_calibration_curves(synthetic_data, fold="fold")
    assert len(fig.data) == 1 + synthetic_data["fold"].nunique()  # diagonal + one curve per fold


def test_confounding_evidence_categories(synthetic_data):
    df = synthetic_data.assign(code=np.resize([10, 2, 1, 0], len(synthetic_data)))
    before = df.copy()
    fig = causaleval.plot_confounding_evidence(df, "propensity", "code")

    pd.testing.assert_frame_equal(df, before)
    assert [trace.name for trace in fig.data] == ["0", "1", "2", "10"]
    assert len({trace.marker.color for trace in fig.data}) == 4


def test_metrics_summary_std_row_is_optional(metrics_summary):
    def row_names(summary):
        return list(causaleval.plot_metrics_summary(summary).data[0].cells.values[0])

    assert row_names(metrics_summary) == ["mean", "95% CI", "std"]
    assert row_names(metrics_summary.drop(columns="std")) == ["mean", "95% CI"]
