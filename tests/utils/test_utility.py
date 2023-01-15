# TODO add more tests!
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import DataUtilityLearning, MemcachedConfig, Utility, powerset


@pytest.mark.parametrize("show_warnings", [False, True])
@pytest.mark.parametrize("num_points, num_features", [(4, 4)])
def test_utility_show_warnings(housing_dataset, show_warnings, recwarn):
    class WarningModel:
        def fit(self, x, y):
            warnings.warn("Warning model fit")
            return self

        def predict(self, x):
            warnings.warn("Warning model predict")
            return np.zeros_like(x)

        def score(self, x, y):
            warnings.warn("Warning model score")
            return 0.0

    utility = Utility(
        model=WarningModel(),
        data=housing_dataset,
        enable_cache=False,
        show_warnings=show_warnings,
    )
    utility([0])

    if show_warnings:
        assert len(recwarn) >= 1
    else:
        assert len(recwarn) == 0


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
@pytest.mark.parametrize("training_budget", [2, 10])
def test_data_utility_learning_wrapper(linear_dataset, training_budget):
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scoring="r2",
        enable_cache=False,
    )
    wrapped_u = DataUtilityLearning(u, training_budget, LinearRegression())
    subsets = list(powerset(wrapped_u.utility.data.indices))

    for s in subsets:
        wrapped_u(s)

    assert len(wrapped_u._utility_samples) == min(len(subsets), training_budget)


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
def test_cache(linear_dataset, memcache_client_config):
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, time_threshold=0
        ),
    )
    subsets = list(powerset(u.data.indices))

    for s in subsets:
        u(s)
    assert u._utility_wrapper.stats.hits == 0

    for s in subsets:
        u(s)
    assert u._utility_wrapper.stats.hits == len(subsets)


@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
def test_different_cache(linear_dataset, memcache_client_config):
    u1 = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, time_threshold=0
        ),
    )
    u2 = Utility(
        model=LinearRegression(fit_intercept=False),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, time_threshold=0
        ),
    )

    assert u1.signature != u2.signature
