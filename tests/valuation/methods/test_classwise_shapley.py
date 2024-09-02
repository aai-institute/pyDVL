from __future__ import annotations

import logging

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from typing_extensions import Self

from pydvl.utils.dataset import Dataset as OldDataset
from pydvl.utils.utility import Utility as OldUtility
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods import ClasswiseShapleyValuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import (
    ClasswiseSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    PermutationSampler,
    UniformSampler,
)
from pydvl.valuation.scorers import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import MaxUpdates
from pydvl.valuation.utility import ClasswiseModelUtility
from pydvl.value import MaxChecks
from pydvl.value.shapley.classwise import ClasswiseScorer as OldClasswiseScorer
from pydvl.value.shapley.classwise import compute_classwise_shapley_values
from pydvl.value.shapley.truncated import NoTruncation
from tests.value import check_values

from .. import check_values

log = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def classwise_shapley_exact_solution() -> tuple[dict, ValuationResult, dict]:
    """
    See [classwise.py][pydvl.valuation.methods.classwise] for details of the derivation.
    """
    return (
        {
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    1 / 6 * np.exp(1 / 4),
                    1 / 3 * np.exp(1 / 4),
                    1 / 12 * np.exp(1 / 4) + 1 / 24 * np.exp(1 / 2),
                    1 / 8 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.fixture(scope="function")
def classwise_shapley_exact_solution_normalized(
    classwise_shapley_exact_solution,
) -> tuple[dict, ValuationResult, dict]:
    """
    It additionally normalizes the values using the argument `normalize_values`. See
    [classwise.py][pydvl.valuation.methods.classwise] for details of the derivation.
    """
    values = classwise_shapley_exact_solution[1].values
    label_zero_coefficient = 1 / np.exp(1 / 4)
    label_one_coefficient = 1 / (1 / 3 * np.exp(1 / 4) + 2 / 3 * np.exp(1 / 2))

    return (
        {
            "normalize_values": True,
        },
        ValuationResult(
            values=np.array(
                [
                    values[0] * label_zero_coefficient,
                    values[1] * label_zero_coefficient,
                    values[2] * label_one_coefficient,
                    values[3] * label_one_coefficient,
                ]
            )
        ),
        {"atol": 0.05},
    )


class ClosedFormLinearClassifier:
    def __init__(self):
        self._beta = None

    def fit(self, x: NDArray, y: NDArray) -> Self:
        x = x[:, 0]
        self._beta = np.dot(x, y) / np.dot(x, x)
        return self

    def predict(self, x: NDArray) -> NDArray:
        if self._beta is None:
            raise AttributeError("Model not fitted")

        probs = self._beta * x
        return np.clip(np.round(probs + 1e-10), 0, 1).astype(int)

    def score(self, x: NDArray, y: NDArray) -> float:
        pred_y = self.predict(x)
        return np.sum(pred_y == y) / 4


@pytest.fixture(scope="function")
def classwise_shapley_utility(
    test_dataset_manual_derivation: Dataset,
) -> ClasswiseModelUtility:
    return ClasswiseModelUtility(
        ClosedFormLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", test_dataset_manual_derivation),
        catch_errors=False,
    )


@pytest.fixture(scope="function")
def train_dataset_manual_derivation() -> Dataset:
    """
    See [classwise.py][pydvl.valuation.methods.classwise] for more details.
    """
    x_train = np.arange(1, 5).reshape([-1, 1])
    y_train = np.array([0, 0, 1, 1])
    return Dataset(x_train, y_train)


@pytest.fixture(scope="function")
def test_dataset_manual_derivation(train_dataset_manual_derivation) -> Dataset:
    """
    See [classwise.py][pydvl.valuation.methods.classwise] for more details.
    """
    x_test = train_dataset_manual_derivation.x
    y_test = np.array([0, 0, 0, 1])
    return Dataset(x_test, y_test)


@pytest.mark.parametrize("n_samples", [500], ids=lambda x: "n_samples={}".format(x))
@pytest.mark.parametrize(
    "exact_solution",
    [
        pytest.param("classwise_shapley_exact_solution", marks=[pytest.mark.xfail]),
        "classwise_shapley_exact_solution_normalized",
    ],
)
def test_classwise_shapley(
    classwise_shapley_utility: ClasswiseModelUtility,
    train_dataset_manual_derivation: Dataset,
    exact_solution: tuple[dict, ValuationResult, dict],
    n_samples: int,
    request,
):
    method_kwargs, exact_solution, check_kwargs = request.getfixturevalue(
        exact_solution
    )
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler()
    sampler = ClasswiseSampler(
        in_class=in_class_sampler, out_of_class=out_of_class_sampler
    )
    valuation = ClasswiseShapleyValuation(
        classwise_shapley_utility,
        sampler=sampler,
        is_done=MaxUpdates(n_samples),
        progress=True,
        **method_kwargs,
    )
    valuation.fit(train_dataset_manual_derivation)
    values = valuation.values()
    check_values(values, exact_solution, **check_kwargs)


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("n_samples", [500], ids=lambda x: "n_samples={}".format(x))
def test_old_vs_new(
    n_samples: int,
    seed,
):
    model = LogisticRegression(random_state=seed)
    old_data = OldDataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=seed,
        stratify_by_target=True,
    )
    old_scorer = OldClasswiseScorer("accuracy", initial_label=0)
    old_u = OldUtility(model=model, data=old_data, scorer=old_scorer)
    old_values = compute_classwise_shapley_values(
        old_u,
        done=MaxChecks(n_samples),
        truncation=NoTruncation(),
        done_sample_complements=MaxChecks(1),
        seed=seed,
    )

    new_train_data = Dataset(old_data.x_train, old_data.y_train)
    new_test_data = Dataset(old_data.x_test, old_data.y_test)

    in_class_sampler = PermutationSampler(seed=seed)
    out_of_class_sampler = UniformSampler(seed=seed)
    sampler = ClasswiseSampler(
        in_class=in_class_sampler,
        out_of_class=out_of_class_sampler,
    )
    new_u = ClasswiseModelUtility(
        model,
        ClasswiseSupervisedScorer("accuracy", new_test_data),
        catch_errors=False,
    )
    valuation = ClasswiseShapleyValuation(
        new_u,
        sampler=sampler,
        is_done=MaxUpdates(n_samples),
    )
    valuation.fit(new_train_data)
    check_values(valuation.values(), old_values, atol=1e-1, rtol=1e-1)
