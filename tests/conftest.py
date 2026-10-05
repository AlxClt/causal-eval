import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_data():
    """
    Observational data where treatment depends on age and sex through a known logistic model.
    'propensity' is the true propensity, so weighting by 'ip_weight' removes the confounding.
    """
    rng = np.random.default_rng(0)
    n = 5000
    age = rng.uniform(25, 75, n)
    sex = rng.integers(0, 2, n)
    education = rng.choice(["low", "mid", "high"], n)

    propensity = 1 / (1 + np.exp(-(-3 + 0.05 * age + 0.8 * sex)))
    treated = rng.binomial(1, propensity)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "education": education,
        "treated": treated,
        "propensity": propensity,
        "ip_weight": treated / propensity + (1 - treated) / (1 - propensity),
        "fold": np.arange(n) % 5,
    })


@pytest.fixture
def metrics_summary():
    return pd.DataFrame({"mean": [0.62, 0.18], "ci_low": [0.57, 0.17], "ci_high": [0.67, 0.19], "std": [0.04, 0.01]},
                        index=["roc_auc", "brier"])
