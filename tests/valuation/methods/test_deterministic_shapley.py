"""Test the deterministic Shapley valuation methods (combinatorial and permutation)."""

import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pydvl.valuation.dataset import GroupedDataset
from pydvl.valuation.methods import ShapleyValuation
from pydvl.valuation.samplers import (
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
)
from pydvl.valuation.scorers import SupervisedScorer
from pydvl.valuation.stopping import NoStopping
from pydvl.valuation.utility import ModelUtility

from .. import check_total_value, check_values

log = logging.getLogger(__name__)

SAMPLER_CLASSES = [DeterministicPermutationSampler, DeterministicUniformSampler]


@pytest.mark.parametrize(
    "test_game, rtol, total_atol",
    [
        (("symmetric-voting", {"n_players": 4}), 0.1, 1e-5),
        (("shoes", {"left": 2, "right": 1}), 0.1, 1e-5),
        (("shoes", {"left": 1, "right": 2}), 0.1, 1e-5),
        (("shoes", {"left": 1, "right": 1}), 0.1, 1e-5),
        (("shoes", {"left": 2, "right": 4}), 0.1, 1e-5),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize("sampler_class", SAMPLER_CLASSES)
def test_games(sampler_class, test_game, rtol, total_atol):
    valuation = ShapleyValuation(
        utility=test_game.u,
        sampler=sampler_class(),
        progress=False,
        is_done=NoStopping(),
    )
    valuation.fit(test_game.data)
    got = valuation.result
    expected = test_game.shapley_values()
    check_total_value(test_game.u.with_dataset(test_game.data), got, atol=total_atol)
    check_values(got, expected, rtol=rtol)


@pytest.mark.parametrize(
    "a, b, num_points, num_groups, scorer_name",
    [(2, 0, 50, 3, "r2"), (2, 1, 100, 5, "r2"), (2, 1, 100, 5, "explained_variance")],
)
def test_grouped_linear(
    linear_dataset,
    num_groups,
    scorer_name,
    cache_backend,
    rtol=0.01,
    total_atol=1e-5,
):
    # assign groups recursively
    data_train, data_test = linear_dataset
    data_groups = np.random.randint(0, num_groups, len(data_train))

    scorer = SupervisedScorer(scorer_name, data_test, default=0)

    grouped_dataset = GroupedDataset.from_dataset(data_train, data_groups)
    grouped_utility = ModelUtility(
        LinearRegression(),
        scorer=scorer,
        cache_backend=cache_backend,
    )

    valuation_combinatorial = ShapleyValuation(
        utility=grouped_utility,
        sampler=DeterministicUniformSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    valuation_combinatorial.fit(grouped_dataset)
    values_combinatorial = valuation_combinatorial.result

    check_total_value(
        grouped_utility.with_dataset(grouped_dataset),
        values_combinatorial,
        atol=total_atol,
    )

    valuation_permutation = ShapleyValuation(
        utility=grouped_utility,
        sampler=DeterministicPermutationSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    valuation_permutation.fit(grouped_dataset)
    values_permutation = valuation_permutation.result

    check_total_value(
        grouped_utility.with_dataset(grouped_dataset),
        values_permutation,
        atol=total_atol,
    )

    check_values(values_combinatorial, values_permutation, rtol=rtol)


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, num_points, scorer_name",
    [
        (2, 1, 20, "explained_variance"),
        (2, 0, 20, "r2"),
        (2, 1, 20, "neg_median_absolute_error"),
        (2, 1, 20, "r2"),
    ],
)
def test_linear_with_outlier(
    linear_dataset, scorer_name, cache_backend, total_atol=1e-5
):
    data_train, data_test = linear_dataset
    scorer = SupervisedScorer(scorer_name, data_test, default=0)

    outlier_idx = np.random.randint(len(data_train))
    data_train.data().y[outlier_idx] -= 100

    utility = ModelUtility(
        LinearRegression(),
        scorer=scorer,
        cache_backend=cache_backend,
    )

    valuation_permutation = ShapleyValuation(
        utility=utility,
        sampler=DeterministicPermutationSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    result = valuation_permutation.fit(data_train).result.sort()

    check_total_value(utility.with_dataset(data_train), result, atol=total_atol)

    assert result.indices[0] == outlier_idx


@pytest.mark.parametrize(
    "coefficients, scorer_name",
    [
        (np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=3), "neg_median_absolute_error"),
        (np.random.randint(-3, 3, size=3), "explained_variance"),
    ],
)
def test_polynomial(
    polynomial_dataset,
    polynomial_pipeline,
    scorer_name,
    rtol=0.01,
    total_atol=1e-5,
):
    (data_train, data_test), _ = polynomial_dataset

    scorer = SupervisedScorer(scorer_name, data_test, default=0)

    utility = ModelUtility(
        polynomial_pipeline,
        scorer=scorer,
    )

    valuation_combinatorial = ShapleyValuation(
        utility=utility,
        sampler=DeterministicUniformSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    valuation_combinatorial.fit(data_train)
    values_combinatorial = valuation_combinatorial.result

    check_total_value(
        utility.with_dataset(data_train), values_combinatorial, atol=total_atol
    )

    valuation_permutation = ShapleyValuation(
        utility=utility,
        sampler=DeterministicPermutationSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    valuation_permutation.fit(data_train)
    values_permutation = valuation_permutation.result

    check_total_value(
        utility.with_dataset(data_train), values_permutation, atol=total_atol
    )
    check_values(values_combinatorial, values_permutation, rtol=rtol)


@pytest.mark.slow
@pytest.mark.parametrize(
    "coefficients, scorer_name",
    [
        (np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=3), "neg_median_absolute_error"),
        (np.random.randint(-3, 3, size=3), "explained_variance"),
    ],
)
def test_polynomial_with_outlier(
    polynomial_dataset,
    polynomial_pipeline,
    scorer_name,
    cache_backend,
    total_atol=1e-5,
):
    (data_train, data_test), _ = polynomial_dataset
    outlier_idx = np.random.randint(len(data_train))
    data_train.data().y[outlier_idx] *= 100

    scorer = SupervisedScorer(scorer_name, data_test, default=0)

    poly_utility = ModelUtility(
        polynomial_pipeline,
        scorer=scorer,
        cache_backend=cache_backend,
    )

    valuation = ShapleyValuation(
        utility=poly_utility,
        sampler=DeterministicPermutationSampler(),
        progress=False,
        is_done=NoStopping(),
    )

    result = valuation.fit(data_train).result

    check_total_value(poly_utility.with_dataset(data_train), result, atol=total_atol)

    assert result.get(0).idx == outlier_idx
