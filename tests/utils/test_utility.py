# TODO add more tests!

import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import DataUtilityLearning, MemcachedConfig, Utility, powerset


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
