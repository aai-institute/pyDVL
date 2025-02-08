import logging
from copy import deepcopy

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.linear_model import LinearRegression

from pydvl.utils.numeric import num_samples_permutation_hoeffding
from pydvl.utils.status import Status
from pydvl.valuation.dataset import GroupedDataset
from pydvl.valuation.methods import GroupTestingShapleyValuation, ShapleyValuation
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    DeterministicUniformSampler,
    OwenSampler,
    PermutationSampler,
    UniformSampler,
)
from pydvl.valuation.scorers import SupervisedScorer, compose_score, sigmoid
from pydvl.valuation.stopping import MaxChecks, MaxUpdates, NoStopping
from pydvl.valuation.utility import ModelUtility

from .. import check_rank_correlation, check_total_value, check_values

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 6}),
        ("shoes", {"left": 3, "right": 4}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_factory, valuation_class, valuation_kwargs, rtol, atol",
    [
        (
            lambda s: PermutationSampler(seed=s),
            DataShapleyValuation,
            {"is_done": MaxUpdates(500)},
            0.2,
            1e-4,
        ),
        (
            lambda s: UniformSampler(seed=s),
            DataShapleyValuation,
            {"is_done": MaxUpdates(2**10)},
            0.2,
            1e-4,
        ),
        (
            lambda s: OwenSampler(200, n_samples_inner=5, seed=s),
            OwenShapleyValuation,
            {},
            0.2,
            1e-4,
        ),
        (
            lambda s: AntitheticOwenSampler(200, n_samples_inner=5, seed=s),
            OwenShapleyValuation,
            {},
            0.1,
            1e-4,
        ),
        # Because of the inaccuracy of GroupTesting, a high atol is required for the
        # value 0, for which the rtol has no effect.
        (
            None,
            GroupTestingShapleyValuation,
            {"n_samples": 5e4, "epsilon": 0.2},
            0.1,
            1e-2,
        ),
    ],
)
@pytest.mark.flaky(reruns=2)
def test_games(
    test_game,
    n_jobs,
    sampler_factory,
    valuation_class,
    valuation_kwargs,
    rtol,
    atol,
    seed,
):
    """Tests shapley values for all methods using toy games.

    For permutation, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.

    TODO:
        - Uncomment the other methods once they are implemented

    FIXME:
     - We don't have a bound for Owen.
    NOTE:
     - The variance in the combinatorial method is huge, so we need lots of
       samples

    """
    if sampler_factory is not None:
        sampler = sampler_factory(seed)
        valuation = valuation_class(
            utility=test_game.u,
            sampler=sampler,
            progress=False,
            **valuation_kwargs,
        )
    else:
        valuation = valuation_class(
            utility=test_game.u,
            progress=False,
            **valuation_kwargs,
        )
    with parallel_config(n_jobs=n_jobs):
        valuation.fit(test_game.data)
    got = valuation.values()
    expected = test_game.shapley_values()
    check_total_value(test_game.u.with_dataset(test_game.data), got, atol=atol)

    check_values(got, expected, rtol=rtol, atol=atol)


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 12}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler_class, sampler_kwargs, valuation_class, valuation_kwargs",
    [
        # TODO Add Permutation Montecarlo once issue #416 is closed.
        (PermutationSampler, {}, DataShapleyValuation, {"is_done": MaxChecks(50)}),
        (UniformSampler, {}, DataShapleyValuation, {"is_done": MaxChecks(50)}),
        (
            OwenSampler,
            {"n_samples_outer": 20, "n_samples_inner": 4},
            OwenShapleyValuation,
            {},
        ),
        (
            AntitheticOwenSampler,
            {"n_samples_outer": 20, "n_samples_inner": 4},
            OwenShapleyValuation,
            {},
        ),
        (
            None,
            {},
            GroupTestingShapleyValuation,
            {"n_samples": 21, "epsilon": 0.2},
        ),
    ],
)
def test_seed(
    test_game,
    sampler_class,
    sampler_kwargs,
    valuation_class,
    valuation_kwargs,
    seed,
    seed_alt,
):
    values = []
    for s in [seed, seed, seed_alt]:
        if sampler_class is not None:
            valuation = valuation_class(
                utility=test_game.u,
                sampler=sampler_class(seed=s, **sampler_kwargs),
                progress=False,
                # TODO: Why is a deepcopy necessary here?
                **deepcopy(valuation_kwargs),
            )
        else:
            valuation = valuation_class(
                utility=test_game.u,
                progress=False,
                **valuation_kwargs,
                seed=s,
            )
        valuation.fit(test_game.data)
        values.append(valuation.values())

    values_1, values_2, values_3 = values

    np.testing.assert_equal(values_1.values, values_2.values)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(values_1.values, values_3.values)


@pytest.mark.slow
@pytest.mark.parametrize("num_samples, delta, eps", [(6, 0.1, 0.1)])
@pytest.mark.parametrize(
    "sampler_class",
    [
        PermutationSampler,
        UniformSampler,
    ],
)
@pytest.mark.flaky(reruns=1)
def test_hoeffding_bound_montecarlo(
    analytic_shapley,
    dummy_train_data,
    n_jobs,
    sampler_class,
    delta,
    eps,
):
    u, exact_values = analytic_shapley

    n_samples = num_samples_permutation_hoeffding(delta=delta, eps=eps, u_range=1)

    for _ in range(10):
        sampler = sampler_class()
        valuation = DataShapleyValuation(
            utility=u,
            sampler=sampler,
            progress=False,
            is_done=MaxChecks(n_samples),
        )
        with parallel_config(n_jobs=n_jobs):
            valuation.fit(dummy_train_data)
        values = valuation.values()

        check_total_value(u, values, atol=len(dummy_train_data) * eps)
        check_rank_correlation(values, exact_values, threshold=0.8)


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, num_points",
    [(2, 0, 21)],  # training set will have 0.3 * 21 ~= 6 samples
)
@pytest.mark.parametrize(
    "sampler_class, sampler_kwargs, valuation_class, valuation_kwargs",
    [
        (PermutationSampler, {}, DataShapleyValuation, {"is_done": MaxUpdates(500)}),
        (
            OwenSampler,
            {"n_samples_outer": 400, "n_samples_inner": 4},
            OwenShapleyValuation,
            {},
        ),
        (
            AntitheticOwenSampler,
            {"n_samples_outer": 400, "n_samples_inner": 4},
            OwenShapleyValuation,
            {},
        ),
        (
            None,
            {},
            GroupTestingShapleyValuation,
            {"n_samples": 5e4, "epsilon": 0.25},
        ),
    ],
)
def test_linear_montecarlo_with_outlier(
    linear_dataset,
    n_jobs,
    sampler_class,
    sampler_kwargs,
    valuation_class,
    valuation_kwargs,
    cache_backend,
):
    """Tests whether valuation methods are able to detect an obvious outlier.

    A point is selected at random from a linear dataset and the dependent
    variable is set to 10 standard deviations.

    Note that this implies that the whole dataset will have very low utility:
    e.g. for R^2 it will be very negative. The larger the range of the utility,
    the more samples are required for the Monte Carlo approximations to converge,
    as indicated by the Hoeffding bound.
    """
    data_train, data_test = linear_dataset

    scorer = compose_score(
        SupervisedScorer("r2", data_test, default=-np.inf),
        sigmoid,
        name="squashed r2",
    )

    outlier_idx = np.random.randint(len(data_train))
    data_train.data().y[outlier_idx] -= 100

    utility = ModelUtility(
        LinearRegression(),
        scorer=scorer,
        cache_backend=cache_backend,
    )

    if sampler_class is not None:
        valuation = valuation_class(
            utility=utility,
            sampler=sampler_class(**sampler_kwargs),
            progress=False,
            **valuation_kwargs,
        )
    else:
        valuation = valuation_class(
            utility=utility,
            progress=False,
            **valuation_kwargs,
        )

    with parallel_config(n_jobs=n_jobs):
        valuation.fit(data_train)
    values = valuation.values()
    values.sort()

    assert values.status == Status.Converged
    assert values[0].index == outlier_idx

    check_total_value(utility.with_dataset(data_train), values, atol=0.2)


@pytest.mark.parametrize(
    "a, b, num_points, num_groups",
    [(2, 0, 21, 2)],  # 24*0.3=6 samples in 2 groups
)
@pytest.mark.parametrize(
    "sampler_class, kwargs",
    [
        (PermutationSampler, dict(is_done=MaxUpdates(700))),
    ],
)
def test_grouped_linear_montecarlo_shapley(
    linear_dataset,
    n_jobs,
    num_groups: int,
    sampler_class,
    kwargs: dict,
    cache_backend,
):
    """
    For permutation and truncated montecarlo, the rtol for each scorer is chosen
    so that the number of samples selected is just above the (ε,δ) bound for ε =
    rtol, δ=0.001 and the range corresponding to each score. This means that
    roughly once every 1000/num_methods runs the test will fail.
    """
    data_train, data_test = linear_dataset

    scorer = compose_score(
        SupervisedScorer("r2", data_test, default=-np.inf),
        sigmoid,
        name="squashed r2",
    )

    rtol = 0.1

    data_groups = np.random.randint(0, num_groups, len(data_train))
    grouped_linear_dataset = GroupedDataset.from_dataset(data_train, data_groups)
    utility = ModelUtility(
        LinearRegression(),
        scorer=scorer,
        cache_backend=cache_backend,
    )

    valuation = DataShapleyValuation(
        utility=utility,
        sampler=sampler_class(),
        progress=False,
        **kwargs,
    )
    with parallel_config(n_jobs=n_jobs):
        valuation.fit(grouped_linear_dataset)
    values = valuation.values()

    exact_valuation = DataShapleyValuation(
        utility=utility,
        sampler=DeterministicUniformSampler(),
        progress=False,
        is_done=NoStopping(),
    )
    exact_valuation.fit(grouped_linear_dataset)
    exact_values = exact_valuation.values()

    check_values(values, exact_values, rtol=rtol)
