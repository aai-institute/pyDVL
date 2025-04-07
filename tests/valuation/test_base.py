import os
import tempfile
from pathlib import Path

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
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.classwise import ClasswiseSampler
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.utility.classwise import ClasswiseModelUtility

from . import recursive_make


class TestValuation(Valuation):
    def fit(self, data=None):
        self.result = ValuationResult.from_random(size=10)
        return self


def test_save_and_load_file_path():
    """Test saving and loading using a file path."""
    valuation = TestValuation().fit()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            valuation.save_result(temp_path)
            loaded_valuation = TestValuation().load_result(temp_path)
            assert loaded_valuation.is_fitted
            assert loaded_valuation.values() == valuation.values()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def test_save_and_load_path_object():
    """Test saving and loading using a Path object."""
    valuation = TestValuation().fit()

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_path = Path(temp_dir) / "subdir" / "test.result"
            valuation.save_result(temp_path)
            assert temp_path.exists()
            loaded_valuation = TestValuation().load_result(temp_path)
            assert loaded_valuation.is_fitted
            assert loaded_valuation.values() == valuation.values()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def test_load_nonexistent_file():
    non_existent_file = "this_file_does_not_exist.pkl"
    valuation = TestValuation().load_result(non_existent_file, ignore_missing=True)
    assert valuation.result is None
    assert not valuation.is_fitted

    with pytest.raises(
        FileNotFoundError, match=f"File '{non_existent_file}' not found"
    ):
        TestValuation().load_result(non_existent_file, ignore_missing=False)


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
    train, test = create_classification_dataset(n_samples=12, random_state=seed)

    model_utility = (
        ModelUtility,
        {
            "model": log_regression,
            "scorer": (
                SupervisedScorer,
                {
                    "scoring": "accuracy",
                    "test_data": test,
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
                    "model": lambda: BaggingClassifier(
                        estimator=log_regression(seed),
                        n_estimators=10,
                        random_state=seed,
                        max_samples=0.5,
                    ).fit(*test.data())
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
                                    "test_data": test,
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
                    "test_data": test,
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

    valuation = recursive_make(params["valuation_cls"], params["valuation_kwargs"])

    result = valuation.init_or_check_result(train)

    assert isinstance(result, ValuationResult)
    assert result.algorithm == str(valuation)
    assert len(result) == len(train)
    assert np.array_equal(result.indices, train.indices)
    assert np.array_equal(result.names, np.array([str(i) for i in train.indices]))
    assert np.all(result.values == 0)
    assert np.all(result.variances == 0)
    assert np.all(result.counts == 0)

    valuation.result = result
    valuation._restored_result = True  # simulate restored result from file
    result2 = valuation.init_or_check_result(train)
    assert result2 is result


@pytest.mark.parametrize("params", valuation_methods())
def test_init_result_mismatch(params, seed):
    """Test that init_result raises errors on mismatched indices or names"""
    # Create datasets
    train1, test1 = create_classification_dataset(n_samples=8, random_state=seed)
    train2, test2 = create_classification_dataset(n_samples=6, random_state=seed)

    # Create valuation object and initialize with data1
    valuation = recursive_make(params["valuation_cls"], params["valuation_kwargs"])
    valuation.result = ValuationResult.zeros(
        algorithm=str(valuation),
        indices=train1.indices,
        data_names=np.array([str(i) for i in train1.indices]),
    )
    valuation._restored_result = True  # simulate restored result from file
    with pytest.raises(ValueError, match="Either the indices or the names"):
        valuation.init_or_check_result(train2)

    # Test with matching indices but different names
    altered_names = np.array([f"alt_{i}" for i in train1.indices])
    valuation.result = ValuationResult.zeros(
        algorithm=str(valuation),
        indices=train1.indices,
        data_names=altered_names,
    )

    with pytest.raises(ValueError, match="Either the indices or the names"):
        valuation.init_or_check_result(train1)


@pytest.mark.parametrize("params", valuation_methods())
def test_save_load_with_init_result(params, seed):
    """Test init_result in conjunction with save and load methods"""

    train, test = create_classification_dataset(n_samples=8, random_state=seed)

    valuation = recursive_make(params["valuation_cls"], params["valuation_kwargs"])
    valuation.fit(train)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

        try:
            valuation.save_result(temp_path)
            new_valuation = recursive_make(
                params["valuation_cls"], params["valuation_kwargs"]
            )
            new_valuation.load_result(temp_path)

            assert new_valuation.is_fitted
            assert new_valuation.result == valuation.result

            new_valuation.init_or_check_result(train)  # Test with the same dataset
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
