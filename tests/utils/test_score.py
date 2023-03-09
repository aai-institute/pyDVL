from typing import Dict, Tuple, cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from pydvl.utils import Utility, powerset
from pydvl.utils.score import (
    ClasswiseScorer,
    Scorer,
    compose_score,
    squashed_r2,
    squashed_variance,
)
from tests.misc import ThresholdClassifier

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class FittedLinearModel:
    def __init__(self, coef: NDArray):
        self.coef = coef

    def predict(self, X):
        return X @ self.coef

    def fit(self, X, y):
        pass

    def score(self, X, y):
        return np.linalg.norm(X @ self.coef - y)


def test_scorer():
    """Tests the Scorer class."""
    scorer = Scorer("r2")
    assert str(scorer) == "r2"
    assert repr(scorer) == "R2 (scorer=make_scorer(r2_score))"

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert 1.0 == scorer(model, X, X @ coef)


def test_compose_score():
    """Tests the compose_score function."""
    composed = compose_score(Scorer("r2"), sigmoid, (0, 1), "squashed r2")
    assert str(composed) == "squashed r2"
    assert repr(composed) == "SquashedR2 (scorer=r2)"

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == composed(model, X, X @ coef)


def test_squashed_r2():
    """Tests the squashed_r2 scorer."""
    assert str(squashed_r2) == "squashed r2"
    assert repr(squashed_r2) == "SquashedR2 (scorer=r2)"
    assert np.allclose(squashed_r2.range, (0, 1))

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == squashed_r2(model, X, X @ coef)


def test_squashed_variance():
    """Tests the squashed_variance scorer."""
    assert str(squashed_variance) == "squashed explained variance"
    assert (
        repr(squashed_variance)
        == "SquashedExplainedVariance (scorer=explained_variance)"
    )
    assert np.allclose(squashed_variance.range, (0, 1))

    coef = np.array([1, 2])
    X = np.array([[1, 2], [3, 4]])
    model = FittedLinearModel(coef)
    assert sigmoid(1.0) == squashed_variance(model, X, X @ coef)


@pytest.mark.parametrize(
    "dataset_alt_seq_simple",
    [((101, 0.3, 0.4))],
    indirect=True,
)
def test_cs_scorer_on_dataset_alt_seq_simple(dataset_alt_seq_simple):
    """
    Tests the class wise scorer.
    """

    scorer = ClasswiseScorer("accuracy", initial_label=0)
    assert str(scorer) == "classwise accuracy"
    assert repr(scorer) == "ClasswiseAccuracy (scorer=make_scorer(accuracy_score))"

    x, y, info = dataset_alt_seq_simple
    n_element = len(x)
    target_in_cls_acc_0 = (info["left_margin"] * 100 + 1) / n_element
    target_out_of_cls_acc_0 = (info["right_margin"] * 100 + 1) / n_element

    model = ThresholdClassifier()
    in_cls_acc_0, out_of_cls_acc_0 = scorer.estimate_in_cls_and_out_of_cls_score(
        model, x, y
    )
    assert np.isclose(in_cls_acc_0, target_in_cls_acc_0)
    assert np.isclose(out_of_cls_acc_0, target_out_of_cls_acc_0)

    scorer.label = 1
    in_cls_acc_1, out_of_cls_acc_1 = scorer.estimate_in_cls_and_out_of_cls_score(
        model, x, y
    )
    assert in_cls_acc_1 == out_of_cls_acc_0
    assert in_cls_acc_0 == out_of_cls_acc_1

    scorer.label = 0
    value = scorer(model, x, y)
    assert np.isclose(value, in_cls_acc_0 * np.exp(out_of_cls_acc_0))

    scorer.label = 1
    value = scorer(model, x, y)
    assert np.isclose(value, in_cls_acc_1 * np.exp(out_of_cls_acc_1))


def test_cs_scorer_on_alt_seq_cf_linear_classifier_cs_score(
    linear_classifier_cs_scorer: Utility,
):
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
    model = linear_classifier_cs_scorer.model
    scorer = cast(ClasswiseScorer, linear_classifier_cs_scorer.scorer)
    scorer.label = 0

    for set_zero_idx in range(len(subsets_zero)):
        for set_one_idx in range(len(subsets_one)):
            indices = list(subsets_zero[set_zero_idx] + subsets_one[set_one_idx])
            (
                x_train,
                y_train,
            ) = linear_classifier_cs_scorer.data.get_training_data(indices)
            linear_classifier_cs_scorer.model.fit(x_train, y_train)
            fitted_beta = linear_classifier_cs_scorer.model._beta  # noqa
            target_beta = target_betas.iloc[set_zero_idx, set_one_idx]
            assert (
                np.isnan(fitted_beta)
                if np.isnan(target_beta)
                else fitted_beta == target_beta
            )

            (
                x_test,
                y_test,
            ) = linear_classifier_cs_scorer.data.get_test_data()
            in_cls_acc_0, in_cls_acc_1 = scorer.estimate_in_cls_and_out_of_cls_score(
                model, x_test, y_test
            )
            assert (
                in_cls_acc_0 == target_accuracies_zero.iloc[set_zero_idx, set_one_idx]
            )
            assert in_cls_acc_1 == target_accuracies_one.iloc[set_zero_idx, set_one_idx]
