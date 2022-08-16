# TODO add more tests!

import pytest
from sklearn.linear_model import LinearRegression

from valuation.utils import MemcachedConfig, Utility, powerset


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize(
    "a, b, num_points",
    [
        (2, 0, 8),
    ],
)
def test_cache(linear_dataset, memcache_client_config):
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, cache_threshold=0
        ),
    )
    subsets = list(powerset(u.data.indices))

    for s in subsets:
        u(s)
    assert u._utility_wrapper.cache_info.hits == 0

    for s in subsets:
        u(s)
    assert u._utility_wrapper.cache_info.hits == len(subsets)


@pytest.mark.parametrize(
    "a, b, num_points",
    [
        (2, 0, 8),
    ],
)
def test_different_cache(linear_dataset, memcache_client_config):
    u1 = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, cache_threshold=0
        ),
    )
    u2 = Utility(
        model=LinearRegression(fit_intercept=False),
        data=linear_dataset,
        scoring="r2",
        enable_cache=True,
        cache_options=MemcachedConfig(
            client_config=memcache_client_config, cache_threshold=0
        ),
    )

    assert u1._utility_wrapper._signature != u2._utility_wrapper._signature
