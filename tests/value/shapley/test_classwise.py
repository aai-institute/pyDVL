"""
TestCases:

1 Not fitting utility function

"""
from typing import Tuple

import numpy as np
import pytest
from numpy._typing import NDArray

from pydvl.utils import SupervisedModel
from pydvl.value.shapley.classwise import _estimate_in_out_cls_accuracy


@pytest.fixture(scope="function")
def mock_model() -> SupervisedModel:
    class _MockModel(SupervisedModel):
        def fit(self, x: NDArray, y: NDArray) -> float:
            raise NotImplementedError("Mock model")

        def predict(self, x: NDArray) -> NDArray:
            y = 0.5 < x
            return y[:, 0].astype(int)

        def score(self, x: NDArray, y: NDArray) -> float:
            raise NotImplementedError("Mock model")

    return _MockModel()


@pytest.mark.parametrize("n_element, left_margin, right_margin", [(101, 0.3, 0.4)])
def test_estimate_in_out_cls_accuracy(
    mock_model: SupervisedModel, n_element: int, left_margin: float, right_margin: float
):
    """
    Simple test case with a mocked model for the in and out of class accuracies. The label set
    is given in the form of 0000011100011111, where the left and right margin can be manipulated.
    """
    x = np.linspace(0, 1, n_element)
    y = ((left_margin <= x) & (x < 0.5)) | ((1 - right_margin) <= x)
    y = y.astype(int)
    x = np.expand_dims(x, -1)

    in_cls_acc_0, out_of_cls_acc_0 = _estimate_in_out_cls_accuracy(mock_model, x, y, 0)
    assert in_cls_acc_0 == (left_margin * 100 + 1) / n_element
    assert out_of_cls_acc_0 == (right_margin * 100 + 1) / n_element

    in_cls_acc_1, out_of_cls_acc_1 = _estimate_in_out_cls_accuracy(mock_model, x, y, 1)
    assert in_cls_acc_1 == out_of_cls_acc_0
    assert in_cls_acc_0 == out_of_cls_acc_1
