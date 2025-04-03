# TODO add more tests!
import pickle
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.utils import Dataset, DataUtilityLearning, Scorer, Utility, powerset
from pydvl.utils.caching import CachedFuncConfig, InMemoryCacheBackend


@pytest.mark.parametrize("show_warnings", [False, True])
def test_utility_show_warnings(show_warnings, recwarn):
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
        data=Dataset.from_arrays(np.zeros((4, 4)), np.zeros(4)),
        show_warnings=show_warnings,
    )
    utility([0])

    if show_warnings:
        assert len(recwarn) >= 1, recwarn.list
    else:
        assert len(recwarn) == 0, recwarn.list


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
@pytest.mark.parametrize("training_budget", [2, 10])
def test_data_utility_learning_wrapper(linear_dataset, training_budget):
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scorer=Scorer("r2"),
    )
    wrapped_u = DataUtilityLearning(u, training_budget, LinearRegression())
    subsets = list(powerset(wrapped_u.utility.data.indices))

    for s in subsets:
        wrapped_u(s)

    assert len(wrapped_u._utility_samples) == min(len(subsets), training_budget)


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
def test_utility_with_cache(linear_dataset):
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scorer=Scorer("r2"),
        cache_backend=InMemoryCacheBackend(),
        cached_func_options=CachedFuncConfig(time_threshold=0.0),
    )
    subsets = list(powerset(u.data.indices))

    for s in subsets:
        u(s)
    assert u._utility_wrapper.stats.hits == 0, u._utility_wrapper.stats

    for s in subsets:
        u(s)

    assert u._utility_wrapper.stats.hits == len(subsets), u._utility_wrapper.stats


@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
def test_different_utility_with_same_cache(linear_dataset):
    cache_backend = InMemoryCacheBackend()
    u1 = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scorer=Scorer("r2"),
        cache_backend=cache_backend,
        cached_func_options=CachedFuncConfig(time_threshold=0.0),
    )
    u2 = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scorer=Scorer("max_error"),
        cache_backend=cache_backend,
        cached_func_options=CachedFuncConfig(time_threshold=0.0),
    )

    subset = u1.data.indices
    # Call first utility with empty cache
    # We expect a cache miss
    u1(subset)
    assert cache_backend.stats.hits == 0
    assert cache_backend.stats.misses == 1
    assert cache_backend.stats.sets == 1

    # Call first utility again
    # We expect a cache hit
    u1(subset)
    assert cache_backend.stats.hits == 1
    assert cache_backend.stats.misses == 1
    assert cache_backend.stats.sets == 1

    # Call second utility
    # We expect a cache miss
    u2(subset)
    assert cache_backend.stats.hits == 1
    assert cache_backend.stats.misses == 2
    assert cache_backend.stats.sets == 2


@pytest.mark.parametrize("a, b, num_points", [(2, 0, 8)])
@pytest.mark.parametrize("use_cache", [False, True])
def test_utility_serialization(linear_dataset, use_cache):
    if use_cache:
        cache = InMemoryCacheBackend()
    else:
        cache = None
    u = Utility(
        model=LinearRegression(),
        data=linear_dataset,
        scorer=Scorer("r2"),
        cache_backend=cache,
    )
    u_unpickled = pickle.loads(pickle.dumps(u))
    assert type(u.model) is type(u_unpickled.model)
    assert type(u.scorer) is type(u_unpickled.scorer)
    assert type(u.data) is type(u_unpickled.data)
    assert (u.data.x_train == u_unpickled.data.x_train).all()
