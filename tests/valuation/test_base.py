from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from pydvl.valuation import (
    AntitheticSampler,
    KNNShapleyValuation,
    MaxSamples,
    ModelUtility,
    NoIndexIteration,
    PermutationSampler,
    SupervisedScorer,
)
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods.classwise_shapley import ClasswiseShapleyValuation
from pydvl.valuation.methods.data_oob import DataOOBValuation
from pydvl.valuation.methods.least_core import (
    ExactLeastCoreValuation,
    MonteCarloLeastCoreValuation,
)
from pydvl.valuation.result import ValuationResult, load_result, save_result
from pydvl.valuation.samplers.classwise import ClasswiseSampler
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility

from . import recursive_make


class TestValuation(Valuation):
    algorithm_name = "TestValuation"

    def fit(self, data: Dataset, continue_from: ValuationResult | None = None):
        self._result = self._init_or_check_result(data, continue_from)
        self._result += ValuationResult(
            indices=data.indices,
            values=np.ones_like(data.indices),
            counts=np.ones_like(data.indices),
            algorithm=str(self),
        )
        return self


def test_save_and_load_file_path(seed, dummy_train_data):
    """Test saving and loading using a file path."""

    valuation = TestValuation().fit(dummy_train_data)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            save_result(valuation.result, temp_path)
            loaded_result = load_result(temp_path)

            continued_valuation = TestValuation().fit(
                dummy_train_data, continue_from=loaded_result
            )

            assert np.all(continued_valuation.result.counts == 2)

            fresh_valuation = TestValuation().fit(dummy_train_data)
            assert np.all(fresh_valuation.result.counts == 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def test_load_nonexistent_file(dummy_train_data):
    """Test loading a nonexistent file."""
    non_existent_file = "this_file_does_not_exist.pkl"

    # Should return None when ignore_missing=True
    loaded_result = load_result(non_existent_file, ignore_missing=True)
    assert loaded_result is None

    # Should raise error when ignore_missing=False
    with pytest.raises(
        FileNotFoundError, match=f"File '{non_existent_file}' not found"
    ):
        load_result(non_existent_file, ignore_missing=False)

    valuation = TestValuation().fit(dummy_train_data, continue_from=None)
    np.testing.assert_allclose(valuation.result.counts, 1)


def test_continue_from_mismatch():
    """Test that continuing from mismatched indices raises an error."""
    np.random.seed(42)

    # Create two datasets with different sizes
    data1, _ = Dataset.from_arrays(np.random.rand(10, 2), np.random.randint(0, 2, 10))
    data2, _ = Dataset.from_arrays(np.random.rand(15, 2), np.random.randint(0, 2, 15))

    valuation = TestValuation().fit(data1)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            save_result(valuation.result, temp_path)
            loaded_result = load_result(temp_path)

            # Try to continue with mismatched data
            with pytest.raises(ValueError, match="Either the indices or the names"):
                TestValuation().fit(data2, continue_from=loaded_result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def create_classification_dataset(
    n_samples,
    random_state,
    n_features=4,
    n_classes=2,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
    )
    return Dataset.from_arrays(X, y, train_size=0.8, random_state=random_state)


def valuation_methods(seed=42):
    log_regression = lambda rs=seed: LogisticRegression(max_iter=10, random_state=rs)

    model_utility = (
        ModelUtility,
        {
            "model": log_regression,
            "scorer": (
                SupervisedScorer,
                {
                    "scoring": "accuracy",
                    "test_data": lambda d: d,
                    "default": 0,
                    "range": (0, 1),
                },
            ),
        },
    )

    out_of_class_sampler = AntitheticSampler(
        index_iteration=NoIndexIteration, seed=seed
    )
    return [
        pytest.param(
            {
                "valuation_cls": DataOOBValuation,
                "valuation_kwargs": {
                    "model": lambda test_data: BaggingClassifier(
                        estimator=log_regression(seed),
                        n_estimators=10,
                        random_state=seed,
                        max_samples=0.5,
                    ).fit(*test_data.data())
                },
            },
            id="DataOOBValuation",
        ),
        pytest.param(
            {
                "valuation_cls": ClasswiseShapleyValuation,
                "valuation_kwargs": {
                    "utility": (
                        ClasswiseModelUtility,
                        {
                            "model": log_regression,
                            "scorer": (
                                ClasswiseSupervisedScorer,
                                {
                                    "scoring": "accuracy",
                                    "test_data": lambda d: d,
                                    "default": 0,
                                    "range": (0, 1),
                                },
                            ),
                        },
                    ),
                    "sampler": (
                        ClasswiseSampler,
                        {
                            "in_class": PermutationSampler(seed=seed),
                            "out_of_class": out_of_class_sampler,
                            "max_in_class_samples": 1,
                        },
                    ),
                    "is_done": (
                        MaxSamples,
                        {"sampler": out_of_class_sampler, "n_samples": 20},
                    ),
                    "progress": False,
                    "normalize_values": False,
                },
            },
            id="ClasswiseShapleyValuation",
        ),
        pytest.param(
            {
                "valuation_cls": ExactLeastCoreValuation,
                "valuation_kwargs": {
                    "utility": model_utility,
                    "non_negative_subsidy": True,
                    "progress": False,
                },
            },
            id="ExactLeastCoreValuation",
        ),
        pytest.param(
            {
                "valuation_cls": MonteCarloLeastCoreValuation,
                "valuation_kwargs": {
                    "utility": model_utility,
                    "n_samples": 10,
                    "seed": seed,
                    "progress": False,
                },
            },
            id="MonteCarloLeastCoreValuation",
        ),
        pytest.param(
            {
                "valuation_cls": KNNShapleyValuation,
                "valuation_kwargs": {
                    "model": KNeighborsClassifier(n_neighbors=3),
                    "test_data": lambda d: d,
                    "progress": False,
                },
            },
            id="KNNShapleyValuation",
        ),
    ]


@pytest.mark.parametrize("params", valuation_methods())
def test_init_result(params, seed):
    """Test that init_result correctly initializes ValuationResult objects"""
    train, test = create_classification_dataset(n_samples=8, random_state=seed)

    valuation = recursive_make(
        params["valuation_cls"],
        params["valuation_kwargs"],
        test_data=test,
        # this is weird: it's for the constructor of BaggingClassifier above
        model=test,
    )

    result = valuation._init_or_check_result(train)

    np.testing.assert_allclose(result.counts, 0)
    np.testing.assert_allclose(result.values, 0)

    result2 = valuation._init_or_check_result(train, result)
    assert result2 is result
