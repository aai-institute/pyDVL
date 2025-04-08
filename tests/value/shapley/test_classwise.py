from __future__ import annotations

from typing import Dict, Tuple, cast

import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.typing import NDArray
from packaging import version
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from pydvl.utils import Dataset as OldDataset
from pydvl.utils import Utility, powerset
from pydvl.utils import Utility as OldUtility
from pydvl.valuation import (
    ClasswiseModelUtility,
    ClasswiseSampler,
    ClasswiseShapleyValuation,
    ClasswiseSupervisedScorer,
    Dataset,
    NoIndexIteration,
    PermutationSampler,
    UniformSampler,
)
from pydvl.valuation.stopping import MaxUpdates
from pydvl.value import ClasswiseScorer as OldClasswiseScorer
from pydvl.value import MaxChecks, ValuationResult
from pydvl.value.shapley.classwise import (
    ClasswiseScorer,
    compute_classwise_shapley_values,
)
from pydvl.value.shapley.truncated import NoTruncation
from tests.value import check_values


@pytest.fixture(scope="function")
def classwise_shapley_exact_solution() -> Tuple[Dict, ValuationResult, Dict]:
    """
    See [classwise.py][pydvl.value.shapley.classwise] for details of the derivation.
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
) -> Tuple[Dict, ValuationResult, Dict]:
    """
    It additionally normalizes the values using the argument `normalize_values`. See
    [classwise.py][pydvl.value.shapley.classwise] for details of the derivation.
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


@pytest.fixture(scope="function")
def classwise_shapley_exact_solution_no_default() -> Tuple[Dict, ValuationResult, Dict]:
    """
    Note that this special case doesn't set the utility to 0 if the permutation is
    empty. See [classwise.py][pydvl.value.shapley.classwise] for details of the
    derivation.
    """
    return (
        {
            "use_default_scorer_value": False,
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    1 / 24 * np.exp(1 / 4),
                    5 / 24 * np.exp(1 / 4),
                    1 / 12 * np.exp(1 / 4) + 1 / 24 * np.exp(1 / 2),
                    1 / 8 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.fixture(scope="function")
def classwise_shapley_exact_solution_no_default_allow_empty_set() -> Tuple[
    Dict, ValuationResult, Dict
]:
    r"""
    Note that this special case doesn't set the utility to 0 if the permutation is
    empty and additionally allows $S^{(k)} = \emptyset$. See
    [classwise.py][pydvl.value.shapley.classwise] for details of the derivation.
    """
    return (
        {
            "use_default_scorer_value": False,
            "min_elements_per_label": 0,
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    3 / 32 + 1 / 32 * np.exp(1 / 4),
                    3 / 32 + 5 / 32 * np.exp(1 / 4),
                    5 / 32 * np.exp(1 / 4) + 1 / 32 * np.exp(1 / 2),
                    1 / 32 * np.exp(1 / 4) + 3 / 32 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.mark.parametrize("n_samples", [500], ids=lambda x: "n_samples={}".format(x))
@pytest.mark.parametrize(
    "n_resample_complement_sets",
    [1],
    ids=lambda x: "n_resample_complement_sets={}".format(x),
)
@pytest.mark.parametrize(
    "exact_solution",
    [
        "classwise_shapley_exact_solution",
        "classwise_shapley_exact_solution_normalized",
        "classwise_shapley_exact_solution_no_default",
        "classwise_shapley_exact_solution_no_default_allow_empty_set",
    ],
)
def test_classwise_shapley(
    classwise_shapley_utility: Utility,
    exact_solution: Tuple[Dict, ValuationResult, Dict],
    n_samples: int,
    n_resample_complement_sets: int,
    request,
):
    args, exact_solution, check_args = request.getfixturevalue(exact_solution)
    values = compute_classwise_shapley_values(
        classwise_shapley_utility,
        done=MaxChecks(n_samples),
        truncation=NoTruncation(),
        done_sample_complements=MaxChecks(n_resample_complement_sets),
        **args,
        progress=True,
    )
    check_values(values, exact_solution, **check_args)
    assert np.all(values.counts == n_samples * n_resample_complement_sets)


def test_classwise_scorer_representation():
    """
    Tests the (string) representation of the ClassWiseScorer.
    """

    scorer = ClasswiseScorer("accuracy", initial_label=0)
    assert str(scorer) == "classwise accuracy"
    if version.parse(sklearn.__version__) >= version.parse("1.4.0"):
        assert (
            repr(scorer)
            == "ClasswiseAccuracy (scorer=make_scorer(accuracy_score, response_method='predict'))"
        )
    else:
        assert repr(scorer) == "ClasswiseAccuracy (scorer=make_scorer(accuracy_score))"


@pytest.mark.parametrize("n_element, left_margin, right_margin", [(101, 0.3, 0.4)])
def test_classwise_scorer_utility(dataset_left_right_margins):
    """
    Tests whether the ClassWiseScorer returns the expected utility value.
    See [classwise.py][pydvl.value.shapley.classwise] for more details.
    """
    scorer = ClasswiseScorer("accuracy", initial_label=0)
    x, y, info = dataset_left_right_margins
    n_element = len(x)
    target_in_cls_acc_0 = (info["left_margin"] * 100 + 1) / n_element
    target_out_of_cls_acc_0 = (info["right_margin"] * 100 + 1) / n_element

    model = ThresholdClassifier()
    in_cls_acc_0, out_of_cls_acc_0 = scorer.estimate_in_class_and_out_of_class_score(
        model, x, y
    )
    np.testing.assert_allclose(in_cls_acc_0, target_in_cls_acc_0, atol=1e-5)
    np.testing.assert_allclose(out_of_cls_acc_0, target_out_of_cls_acc_0, atol=1e-5)

    value = scorer(model, x, y)
    np.testing.assert_allclose(
        value, in_cls_acc_0 * np.exp(out_of_cls_acc_0), atol=1e-5
    )

    scorer.label = 1
    value = scorer(model, x, y)
    np.testing.assert_allclose(
        value, out_of_cls_acc_0 * np.exp(in_cls_acc_0), atol=1e-5
    )


@pytest.mark.parametrize("n_element, left_margin, right_margin", [(101, 0.3, 0.4)])
def test_classwise_scorer_is_symmetric(
    dataset_left_right_margins,
):
    """
    Tests whether the ClassWiseScorer is symmetric. For a two-class classification the
    in-class accuracy for the first label needs to match the out-of-class accuracy for
    the second label. See [classwise.py][pydvl.value.shapley.classwise] for more
    details.
    """
    scorer = ClasswiseScorer("accuracy", initial_label=0)
    x, y, info = dataset_left_right_margins
    model = ThresholdClassifier()
    in_cls_acc_0, out_of_cls_acc_0 = scorer.estimate_in_class_and_out_of_class_score(
        model, x, y
    )
    scorer.label = 1
    in_cls_acc_1, out_of_cls_acc_1 = scorer.estimate_in_class_and_out_of_class_score(
        model, x, y
    )
    assert in_cls_acc_1 == out_of_cls_acc_0
    assert in_cls_acc_0 == out_of_cls_acc_1


def test_classwise_scorer_accuracies_manual_derivation(
    classwise_shapley_utility: Utility,
):
    """
    Tests whether the model of the scorer is fitted correctly and returns the expected
    in-class and out-of-class accuracies. See
    [classwise.py][pydvl.value.shapley.classwise] for more details.
    """
    subsets_zero = list(powerset(np.array((0, 1))))
    subsets_one = list(powerset(np.array((2, 3))))
    subsets_zero = [tuple(s) for s in subsets_zero]
    subsets_one = [tuple(s) for s in subsets_one]
    target_accuracies_zero = pd.DataFrame(
        [
            [0, 1 / 4, 1 / 4, 1 / 4],
            [3 / 4, 1 / 4, 1 / 2, 1 / 4],
            [3 / 4, 1 / 2, 1 / 2, 1 / 2],
            [3 / 4, 1 / 2, 1 / 2, 1 / 2],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    target_accuracies_one = pd.DataFrame(
        [
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    model = classwise_shapley_utility.model
    scorer = cast(ClasswiseScorer, classwise_shapley_utility.scorer)
    scorer.label = 0

    for set_zero_idx in range(len(subsets_zero)):
        for set_one_idx in range(len(subsets_one)):
            indices = list(subsets_zero[set_zero_idx] + subsets_one[set_one_idx])

            # TODO the powersets subsets_zero, subsets_one contain the empty set, having
            #  this leads to an empty index set with the consequence of undefined
            #  behavior (due to nan values). This is NOT the correct fix, this test
            #  must be completely revised!
            if len(indices) == 0:
                continue

            (
                x_train,
                y_train,
            ) = classwise_shapley_utility.data.get_training_data(indices)
            classwise_shapley_utility.model.fit(x_train, y_train)

            (
                x_test,
                y_test,
            ) = classwise_shapley_utility.data.get_test_data()
            (
                in_cls_acc_0,
                in_cls_acc_1,
            ) = scorer.estimate_in_class_and_out_of_class_score(model, x_test, y_test)
            assert (
                in_cls_acc_0 == target_accuracies_zero.iloc[set_zero_idx, set_one_idx]
            )
            assert in_cls_acc_1 == target_accuracies_one.iloc[set_zero_idx, set_one_idx]


@pytest.mark.parametrize("n_element, left_margin, right_margin", [(101, 0.3, 0.4)])
def test_classwise_scorer_accuracies_left_right_margins(dataset_left_right_margins):
    """
    Tests whether the model of the scorer is fitted correctly and returns the expected
    in-class and out-of-class accuracies. See
    [classwise.py][pydvl.value.shapley.classwise] for more details.
    """
    scorer = ClasswiseScorer("accuracy", initial_label=0)
    x, y, info = dataset_left_right_margins
    n_element = len(x)

    target_in_cls_acc_0 = (info["left_margin"] * 100 + 1) / n_element
    target_out_of_cls_acc_0 = (info["right_margin"] * 100 + 1) / n_element

    model = ThresholdClassifier()
    in_cls_acc_0, out_of_cls_acc_0 = scorer.estimate_in_class_and_out_of_class_score(
        model, x, y
    )
    np.testing.assert_allclose(in_cls_acc_0, target_in_cls_acc_0, atol=1e-5)
    np.testing.assert_allclose(out_of_cls_acc_0, target_out_of_cls_acc_0, atol=1e-5)


def test_closed_form_linear_classifier(
    classwise_shapley_utility: Utility,
):
    """
    Tests whether the model is fitted correctly and contains the right $\beta$
    parameter. See [classwise.py][pydvl.value.shapley.classwise] for more details.
    """
    subsets_zero = list(powerset(np.array((0, 1))))
    subsets_one = list(powerset(np.array((2, 3))))
    subsets_zero = [tuple(s) for s in subsets_zero]
    subsets_one = [tuple(s) for s in subsets_one]
    target_betas = pd.DataFrame(
        [
            [np.nan, 1 / 3, 1 / 4, 7 / 25],
            [0, 3 / 10, 4 / 17, 7 / 26],
            [0, 3 / 13, 1 / 5, 7 / 29],
            [0, 3 / 14, 4 / 21, 7 / 30],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    scorer = cast(ClasswiseScorer, classwise_shapley_utility.scorer)
    scorer.label = 0

    for set_zero_idx in range(len(subsets_zero)):
        for set_one_idx in range(len(subsets_one)):
            indices = list(subsets_zero[set_zero_idx] + subsets_one[set_one_idx])
            (
                x_train,
                y_train,
            ) = classwise_shapley_utility.data.get_training_data(indices)
            classwise_shapley_utility.model.fit(x_train, y_train)
            fitted_beta = classwise_shapley_utility.model._beta  # noqa
            target_beta = target_betas.iloc[set_zero_idx, set_one_idx]
            assert (
                np.isnan(fitted_beta)
                if np.isnan(target_beta)
                else fitted_beta == target_beta
            )


class ThresholdClassifier:
    def fit(self, x: NDArray, y: NDArray) -> float:
        raise NotImplementedError("Mock model")

    def predict(self, x: NDArray) -> NDArray:
        y = 0.5 < x
        return y[:, 0].astype(int)

    def score(self, x: NDArray, y: NDArray) -> float:
        raise NotImplementedError("Mock model")


class ClosedFormLinearClassifier:
    def __init__(self):
        self._beta = None

    def fit(self, x: NDArray, y: NDArray) -> float:
        v = x[:, 0]
        self._beta = np.dot(v, y) / np.dot(v, v)
        return -1

    def predict(self, x: NDArray) -> NDArray:
        if self._beta is None:
            raise AttributeError("Model not fitted")

        x = x[:, 0]
        probs = self._beta * x
        return np.clip(np.round(probs + 1e-10), 0, 1).astype(int)

    def score(self, x: NDArray, y: NDArray) -> float:
        pred_y = self.predict(x)
        return np.sum(pred_y == y) / 4


@pytest.fixture(scope="function")
def classwise_shapley_utility(
    dataset_manual_derivation: OldDataset,
) -> Utility:
    return Utility(
        ClosedFormLinearClassifier(),
        dataset_manual_derivation,
        ClasswiseScorer("accuracy"),
        catch_errors=False,
    )


@pytest.fixture(scope="function")
def dataset_manual_derivation() -> OldDataset:
    """
    See [classwise.py][pydvl.value.shapley.classwise] for more details.
    """
    x_train = np.arange(1, 5).reshape([-1, 1])
    y_train = np.array([0, 0, 1, 1])
    x_test = x_train
    y_test = np.array([0, 0, 0, 1])
    return OldDataset(x_train, y_train, x_test, y_test)


@pytest.fixture(scope="function")
def dataset_left_right_margins(
    n_element: int, left_margin: float, right_margin: float
) -> Tuple[NDArray[np.float64], NDArray[np.int_], Dict[str, float]]:
    """
    The label set is represented as 0000011100011111, with adjustable left and right
    margins. The left margin denotes the percentage of zeros at the beginning, while the
    right margin denotes the percentage of ones at the end. Accuracy can be efficiently
    calculated using a closed-form solution.
    """
    x = np.linspace(0, 1, n_element)
    y = ((left_margin <= x) & (x < 0.5)) | ((1 - right_margin) <= x)
    y = y.astype(int)
    x = np.expand_dims(x, -1)
    return x, y, {"left_margin": left_margin, "right_margin": right_margin}


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
    out_of_class_sampler = UniformSampler(seed=seed, index_iteration=NoIndexIteration)
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
