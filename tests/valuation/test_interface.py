"""Simple test for the public user interface documented in tutorials."""

import os
from contextlib import contextmanager

import joblib
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from pydvl.valuation import (
    IndicatorUtilityModel,
    PowerLawSampleSize,
    RandomSizeIteration,
    StratifiedSampler,
)
from pydvl.valuation.methods import *
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import AntitheticSampler, PermutationSampler
from pydvl.valuation.samplers.truncation import DeviationTruncation, RelativeTruncation
from pydvl.valuation.scorers.supervised import SupervisedScorer
from pydvl.valuation.stopping import HistoryDeviation, MaxUpdates
from pydvl.valuation.utility import DataUtilityLearning, ModelUtility


@pytest.fixture
def utility(iris_data):
    model = LogisticRegression()
    _, test = iris_data
    utility = ModelUtility(
        model, SupervisedScorer(model, test, default=0), catch_errors=True
    )
    return utility


@pytest.fixture
def train_data(iris_data):
    train, _ = iris_data
    return train


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_loo_valuation(train_data, utility, n_jobs):
    valuation = LOOValuation(utility=utility, progress=False)
    with disable_logging():
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_shapley_valuation(train_data, utility, n_jobs):
    valuation = ShapleyValuation(
        utility,
        sampler=PermutationSampler(
            DeviationTruncation(sigmas=1.0, burn_in_fraction=0.1)
        ),
        is_done=MaxUpdates(5),
        progress=False,
    )
    with disable_logging():
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


n_jobs_list = [1]

# FIXME: in the CI pipeline, trying to run multiple jobs with joblib crashes the worker
if not os.getenv("CI"):
    n_jobs_list.append(2)


@pytest.mark.parametrize("n_jobs", n_jobs_list)
def test_beta_shapley_valuation(train_data, utility, n_jobs):
    valuation = BetaShapleyValuation(
        utility,
        sampler=AntitheticSampler(),
        is_done=MaxUpdates(5) | HistoryDeviation(n_steps=50, rtol=0.1),
        alpha=1,
        beta=16,
        progress=False,
    )
    with disable_logging():
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)
    assert got.status is Status.Converged


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_delta_shapley_valuation(train_data, utility, n_jobs):
    valuation = DeltaShapleyValuation(
        utility,
        sampler=StratifiedSampler(
            sample_sizes=PowerLawSampleSize(exponent=-2),
            sample_sizes_iteration=RandomSizeIteration,
        ),
        is_done=MaxUpdates(5),
        progress=False,
    )
    with disable_logging():
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_banzhaf_valuation(train_data, utility, n_jobs):
    val_bzf = BanzhafValuation(
        utility,
        sampler=PermutationSampler(RelativeTruncation(rtol=0.1)),
        is_done=MaxUpdates(5),
        progress=False,
    )
    with joblib.parallel_backend("loky", n_jobs=n_jobs):
        val_bzf.fit(train_data)

    got = val_bzf.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_group_testing_valuation(train_data, utility, n_jobs):
    valuation = GroupTestingShapleyValuation(
        utility,
        n_samples=10,
        progress=False,
        epsilon=0.1,
    )
    with disable_logging():
        with joblib.parallel_config(backend="loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_data_utility_learning(train_data, utility, n_jobs):
    utility_model = IndicatorUtilityModel(
        predictor=LinearRegression(), n_data=len(train_data)
    )
    learned_u = DataUtilityLearning(
        utility=utility, training_budget=3, model=utility_model
    )
    valuation = ShapleyValuation(
        learned_u,
        sampler=PermutationSampler(
            DeviationTruncation(burn_in_fraction=0.1, sigmas=1.0)
        ),
        is_done=MaxUpdates(5),
        progress=False,
    )
    with disable_logging():
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            valuation.fit(train_data)

    got = valuation.result
    assert isinstance(got, ValuationResult)
    assert len(got) == len(train_data)


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """A context manager that disables all logging."""
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
