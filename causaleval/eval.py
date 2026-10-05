import warnings
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score


BLUE, ORANGE, GREEN = '#4C78A8', '#F58518', '#54A24B'
FOLD_COLORS = px.colors.qualitative.Set2


def standardized_difference(data, confounder, treatment, weights=None, mode=None, handle_nan='error', return_absolute=True):

    """
    Binary treatment only, value must be 0 or 1
    df: pd.DataFrame the data
    confounder: is the name of the confounder (=column) of interest
    treatment: is the name of the treatment column
    weights: name of the column containing weigths to be applied
    mode: 'continuous' or 'categorical'
    handle_nan: 'drop' or 'error' or 'dummy'
    """

    assert(mode in [None, 'continuous', 'categorical'])
    assert(handle_nan in ['error', 'drop', 'fill'])

    df = data.copy()

    if mode=='continuous':
        df[confounder] = df[confounder].astype(float)
    if mode=='categorical':
        df[confounder] = df[confounder].astype(str)
    if mode is None:
        warnings.warn \
            ('Using mode=None assumes that the confounder is of the corresponding type: str for categoricals, numeric for continuous (watch out for categories represented by numbers)')
        if is_string_dtype(df[confounder]):
            mode ='categorical'
        if is_numeric_dtype(df[confounder]):
            mode ='continuous'


    df_treated = df.loc[df[treatment] == 1].copy()
    df_control = df.loc[df[treatment] == 0].copy()

    if handle_nan == 'drop':
        df_treated = df_treated.dropna(how='any', subset=[confounder ,],  axis=0)
        df_control = df_control.dropna(how='any', subset=[confounder ,],  axis=0)
    if handle_nan == 'error':
        assert(any(df_control.notna()))
        assert(any(df_treated.notna()))
    if handle_nan == 'fill':
        if mode == 'continuous':
            warnings.warn('Imputing missing values for continuious confounder using the within treatment group mean')
            df_treated[confounder] = df_treated[confounder].fillna(df_treated[confounder].mean())
            df_control[confounder] = df_control[confounder].fillna(df_control[confounder].mean())
        else:
            df_treated[confounder] = df_treated[confounder].fillna('Nan_(filled)')
            df_control[confounder] = df_control[confounder].fillna('Nan_(filled)')

    if mode == 'continuous':
        t = df_treated[confounder].values
        c = df_control[confounder].values
        if weights:
            w_t = df_treated[weights].values
            w_c = df_control[weights].values
        else:
            w_t = None
            w_c = None
        a = (np.average(t, weights=w_t) - np.average(c, weights=w_c))
        b = np.sqrt( 0.5 * (np.cov(t, aweights=w_t ) + np.cov(c, aweights=w_c)) )
        d = a/ b

    if mode == 'categorical':

        categories_to_idx = dict((c, i) for i, c in enumerate(df[confounder].unique()))
        idx_to_categories = dict((i, c) for c, i in categories_to_idx.items())
        k = len(categories_to_idx)

        # matrixes
        T = np.zeros(shape=(k - 1,))
        C = np.zeros(shape=(k - 1,))
        S = np.zeros(shape=(k - 1, k - 1))

        # computing probabilities for treated and control groups
        if weights:
            for i in range(k - 1):
                T[i] = df_treated.loc[df_treated[confounder] == idx_to_categories[i + 1]][weights].sum() / df_treated[
                    weights].sum()  # to reproduce the paper where the first category is dropped, but no reason this shouldn't work dropping the last
                C[i] = df_control.loc[df_control[confounder] == idx_to_categories[i + 1]][weights].sum() / df_control[
                    weights].sum()
        else:
            for i in range(k - 1):
                T[i] = df_treated.loc[df_treated[confounder] == idx_to_categories[i + 1]].shape[0] / df_treated.shape[
                    0]  # to reproduce the paper where the first category is dropped, but no reason this shouldn't work dropping the last
                C[i] = df_control.loc[df_control[confounder] == idx_to_categories[i + 1]].shape[0] / df_control.shape[0]

        for i in range(k - 1):
            for j in range(k - 1):
                if i == j:
                    S[i, j] = 1 / 2 * (T[i] * (1 - T[i]) + C[i] * (1 - C[i]))
                else:
                    S[i, j] = 1 / 2 * (T[i] * T[j] + C[i] * C[j])

                    # standardized diff
        S_inv = np.linalg.inv(S)
        d = np.sqrt(np.transpose(T - C) @ S_inv @ (T - C))

    if return_absolute:
        d = abs(d)

    return d


def compute_standardized_differences(datasets, continuous, categoricals, datasets_labels, weights=None, treatment='treated'):
    standardized_differences = pd.DataFrame(index=continuous + categoricals, columns=datasets_labels)

    for c in categoricals:
        for i, l in enumerate(datasets_labels):
            if weights:
                w = weights[i]
            else:
                w = None
            d = standardized_difference(datasets[i], c, treatment=treatment, weights=w, mode='categorical',
                                        handle_nan='fill')
            standardized_differences.loc[c, l] = d

    for c in continuous:
        for i, l in enumerate(datasets_labels):
            if weights:
                w = weights[i]
            else:
                w = None
            d = standardized_difference(datasets[i], c, treatment=treatment, weights=w, mode='continuous',
                                        handle_nan='drop')
            standardized_differences.loc[c, l] = d

    return standardized_differences


def get_propensity_models_roc_curves(y_true, pred_proba, clip_values=(0.001, 1)):
    ip_weights = np.divide(y_true, pred_proba.clip(min=clip_values[0], max=clip_values[1])) + np.divide((1 - y_true), (
                1 - pred_proba).clip(min=clip_values[0], max=clip_values[1]))

    weights_expected_curve = np.concatenate([pred_proba, 1 - pred_proba])
    treatment_expected_curve = np.concatenate([np.ones_like(pred_proba), np.zeros_like(pred_proba)])
    p_hat_expected_curve = np.concatenate([pred_proba, pred_proba])

    curves = {'vanilla': [y_true, pred_proba, None],
              'IP_weighted': [y_true, pred_proba, ip_weights],
              'Expected': [treatment_expected_curve, p_hat_expected_curve, weights_expected_curve]
              }

    return curves


# ---------------------------------------------------------------------------
# Trace builders: shared by the standalone plots and plot_propensity_evaluation
# ---------------------------------------------------------------------------

def _diagonal_trace(name):
    return go.Scatter(mode='lines', x=[0, 1], y=[0, 1], line=dict(color='black', dash='dot'), name=name)


def _standardized_differences_table(data, continuous, categoricals, treatment, weights):
    labels = ['original', 'weighted'] if weights else ['original']
    std_diffs = compute_standardized_differences([data] * len(labels), continuous, categoricals, labels,
                                                 weights=[None, weights] if weights else None, treatment=treatment)
    return std_diffs.astype(float).sort_values(by='original', ascending=True)


def _standardized_differences_traces(std_diffs):
    styles = {'original': dict(color=ORANGE, symbol='triangle-up'),
              'weighted': dict(color=BLUE, symbol='circle')}
    return [go.Scatter(mode='markers', x=std_diffs[c], y=std_diffs.index, name=c, marker=dict(size=9, **styles[c]))
            for c in std_diffs.columns]


def _calibration_traces(data, treatment, propensity, fold, n_bins, strategy):
    groups = data.groupby(fold) if fold else [(None, data)]
    traces = [_diagonal_trace('perfect calibration')]
    for i, (k, group) in enumerate(groups):
        p_true, p_pred = calibration_curve(group[treatment], group[propensity], n_bins=n_bins, strategy=strategy)
        traces.append(go.Scatter(mode='lines+markers', x=p_pred, y=p_true,
                                 name=f'fold {k}' if fold else 'model',
                                 line=dict(color=FOLD_COLORS[i % len(FOLD_COLORS)])))
    return traces


def _propensity_distribution_traces(data, propensity, treatment, bins):
    xbins = dict(start=0, end=1, size=1 / bins)
    return [go.Histogram(x=data.loc[data[treatment] == value, propensity], name=name, xbins=xbins,
                         marker_color=color, opacity=0.6)
            for value, name, color in [(0, 'control', BLUE), (1, 'treated', ORANGE)]]


def _roc_traces(data, treatment, propensity):
    curves = get_propensity_models_roc_curves(data[treatment].values, data[propensity].values)
    styles = {'vanilla': ('propensity model', BLUE),
              'IP_weighted': ('IP-weighted', ORANGE),
              'Expected': ('expected', GREEN)}

    traces = [_diagonal_trace('random classifier')]
    for key, (y_true, y_score, w) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=w)
        auc = roc_auc_score(y_true, y_score, sample_weight=w)
        name, color = styles[key]
        traces.append(go.Scatter(mode='lines', x=fpr, y=tpr, name=f'{name}, AUC {auc:.3f}', line=dict(color=color)))
    return traces


def _metrics_table_trace(metrics_summary, ci_label):
    # one table column per metric, rows: mean, confidence interval, std (if given)
    rows = ['mean', ci_label] + (['std'] if 'std' in metrics_summary.columns else [])
    columns = [[f'{r.mean:.3f}', f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'] + ([f'{r.std:.3f}'] if 'std' in rows else [])
               for r in metrics_summary.itertuples()]
    return go.Table(header=dict(values=[''] + metrics_summary.index.tolist(), fill_color='#E5ECF6'),
                    cells=dict(values=[rows] + columns, fill_color='white', height=24))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_standardized_differences(data, continuous, categoricals, treatment='treated', weights='ip_weight',
                                  highlight=None, threshold=0.1, **fig_kwargs):
    """
    Covariate balance: absolute standardized mean difference of each covariate, in the original data and,
    if weights (name of a weight column) is given, in the weighted data.
    """
    std_diffs = _standardized_differences_table(data, continuous, categoricals, treatment, weights)

    fig = go.Figure(**fig_kwargs)
    fig.add_traces(_standardized_differences_traces(std_diffs))
    fig.add_vline(x=threshold, line_width=3, line_dash='dash', line_color='green', opacity=0.5)

    for h in highlight or []:
        h_idx = std_diffs.index.get_loc(h)
        fig.add_hrect(y0=h_idx - 0.5, y1=h_idx + 0.5, line_width=0, fillcolor='rgba(26,150,65,0.3)')

    fig.update_layout(title='Covariate balance', xaxis_title='Absolute standardized mean difference')
    return fig


def plot_calibration_curves(data, treatment='treated', propensity='propensity', fold=None, n_bins=10,
                            strategy='quantile', **fig_kwargs):
    """
    Calibration of the propensity model, one curve per value of the fold column if given.
    strategy: 'quantile' (bins with the same number of samples) or 'uniform', see sklearn's calibration_curve
    """
    fig = go.Figure(**fig_kwargs)
    fig.add_traces(_calibration_traces(data, treatment, propensity, fold, n_bins, strategy))
    fig.update_layout(title='Calibration', xaxis_title='Predicted propensity', yaxis_title='Observed treatment rate')
    return fig


def plot_propensity_distribution(data, propensity='propensity', treatment='treated', bins=50, **fig_kwargs):
    fig = go.Figure(**fig_kwargs)
    fig.add_traces(_propensity_distribution_traces(data, propensity, treatment, bins))
    fig.update_layout(title='Propensity distribution', barmode='overlay',
                      xaxis_title='Propensity', yaxis_title='Count')
    return fig


def plot_roc_curves(data, treatment='treated', propensity='propensity', **fig_kwargs):
    """
    Vanilla, IP-weighted and expected ROC curves of the propensity model.
    The IP-weighted curve should be close to the diagonal and the vanilla curve close to the expected one.
    """
    fig = go.Figure(**fig_kwargs)
    fig.add_traces(_roc_traces(data, treatment, propensity))
    fig.update_layout(title='ROC curves', xaxis_title='False positive rate', yaxis_title='True positive rate')
    return fig


def plot_metrics_summary(metrics_summary, ci_label='95% CI', **fig_kwargs):
    """
    Table of the out-of-sample metrics of the propensity model.
    metrics_summary: pd.DataFrame indexed by metric name, with columns 'mean', 'ci_low', 'ci_high'
    and optionally 'std'
    """
    fig = go.Figure(**fig_kwargs)
    fig.add_trace(_metrics_table_trace(metrics_summary, ci_label))
    fig.update_layout(title='Out-of-sample metrics')
    return fig


def _add_panel(fig, traces, row, col, legend, corner):
    for trace in traces:
        trace.legend = legend
        fig.add_trace(trace, row=row, col=col)

    subplot = fig.get_subplot(row, col)
    vertical, horizontal = corner.split('-')
    fig.update_layout({legend: dict(
        x=subplot.xaxis.domain[0 if horizontal == 'left' else 1],
        y=subplot.yaxis.domain[0 if vertical == 'bottom' else 1],
        xanchor=horizontal, yanchor=vertical,
        bgcolor='rgba(255,255,255,0.7)')})


def plot_propensity_evaluation(data, continuous, categoricals, treatment='treated', propensity='propensity',
                               weights='ip_weight', fold=None, n_bins=10, strategy='quantile', bins=50, threshold=0.1,
                               metrics_summary=None, ci_label='95% CI', **layout_kwargs):
    """
    2x2 evaluation of a propensity model (as Fig. 3 in https://arxiv.org/pdf/1906.00442):
    covariate balance, calibration, propensity distribution and ROC curves.
    If metrics_summary is given, a table of the out-of-sample metrics is added below (see plot_metrics_summary).
    """
    titles = ['A. Covariate balance', 'B. Calibration', 'C. Propensity distribution', 'D. ROC curves']
    if metrics_summary is None:
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.12, vertical_spacing=0.12, subplot_titles=titles)
        height = 900
    else:
        fig = make_subplots(rows=3, cols=2, horizontal_spacing=0.12, vertical_spacing=0.09,
                            row_heights=[0.43, 0.43, 0.14],
                            specs=[[{}, {}], [{}, {}], [{'type': 'table', 'colspan': 2}, None]],
                            subplot_titles=titles + ['E. Out-of-sample metrics'])
        height = 1080

    std_diffs = _standardized_differences_table(data, continuous, categoricals, treatment, weights)
    _add_panel(fig, _standardized_differences_traces(std_diffs), 1, 1, 'legend', 'bottom-right')
    _add_panel(fig, _calibration_traces(data, treatment, propensity, fold, n_bins, strategy), 1, 2, 'legend2', 'top-left')
    _add_panel(fig, _propensity_distribution_traces(data, propensity, treatment, bins), 2, 1, 'legend3', 'top-right')
    _add_panel(fig, _roc_traces(data, treatment, propensity), 2, 2, 'legend4', 'bottom-right')

    fig.add_vline(x=threshold, line_width=2, line_dash='dash', line_color='green', opacity=0.5, row=1, col=1)

    for (row, col), (x_title, y_title) in {
        (1, 1): ('Absolute standardized mean difference', None),
        (1, 2): ('Predicted propensity', 'Observed treatment rate'),
        (2, 1): ('Propensity', 'Count'),
        (2, 2): ('False positive rate', 'True positive rate'),
    }.items():
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)

    # added last: add_vline fails on figures that already contain a table trace
    if metrics_summary is not None:
        fig.add_trace(_metrics_table_trace(metrics_summary, ci_label), row=3, col=1)

    # a plain string title is shown centered and in bold
    if isinstance(layout_kwargs.get('title'), str):
        layout_kwargs['title'] = dict(text=f"<b>{layout_kwargs['title']}</b>", x=0.5, xanchor='center')

    fig.update_layout(barmode='overlay', height=height, width=1200, **layout_kwargs)
    return fig


def plot_confounder_distributions(data, confounders, treatment='treated', **kwargs):
    assert (type(confounders) == list)
    assert (treatment in data.columns)
    figs = []
    for c in confounders:
        fig = make_subplots(rows=1, cols=2, column_titles=['Control group', 'Treated group'], shared_xaxes='all', shared_yaxes='all',
                            **kwargs)
        # same bins (bingroup) and proportions instead of counts, so that groups of different sizes compare
        for col, (value, name, color) in enumerate([(0, 'control', BLUE), (1, 'treated', ORANGE)], start=1):
            fig.add_trace(go.Histogram(x=data[data[treatment] == value][c], name=name, marker_color=color,
                                       histnorm='probability', bingroup=1), row=1, col=col)
        fig.update_yaxes(title_text='Proportion', row=1, col=1)
        fig.update_xaxes(categoryorder='category ascending')
        figs.append(fig)
    return figs


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    # helper for plot_confounding_evidence
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def round_to_right_digit(x, floating_numbers=2):

    base_num = math.floor(math.log(1/x, 10))
    if base_num > 0:
        return round(x, base_num+floating_numbers)
    else:
        return round(x, floating_numbers)


def plot_confounding_evidence(data, propensity_col, confounder, categorical=True, bins=10):
    data = data.copy()
    if categorical:
        # categories as strings so that numeric codes get discrete colors, ordered by their original values
        order = [str(v) for v in sorted(data[confounder].dropna().unique())]
        data[confounder] = data[confounder].astype(str)
        fig = px.box(data, x=confounder, y=propensity_col, color=confounder, category_orders={confounder: order})
        fig.update_xaxes(type='category')
        fig.update_layout(showlegend=False)

    else:
        cuts, edges = pd.qcut(data[confounder], bins, retbins=True, labels=False, duplicates='drop')
        if len(edges) - 1 < bins:
            warnings.warn(f'Duplicate quantile values of {confounder} were dropped: {len(edges) - 1} bins instead of {bins}')

        data[f'quantiles of {confounder}'] = cuts
        edges = list(map(lambda x: f'{round_to_right_digit(x[0])} to {round_to_right_digit(x[1])}', pairwise(edges)))
        fig = px.box(data,
                     x=f'quantiles of {confounder}',
                     y=propensity_col,
                     color=f'quantiles of {confounder}'
                     )
        fig.update_xaxes(tickangle=45, tickmode = 'array', tickvals= list(range(len(edges))), ticktext=edges)
        fig.update_layout(showlegend=False)

    return fig
