import numpy as np
import pytest

from valuation.utils import vanishing_derivatives
from valuation.utils.numeric import powerset


def test_dataset_len(boston_dataset):
    assert len(boston_dataset) == len(boston_dataset.x_train) == 404
    assert len(boston_dataset.x_train) + len(boston_dataset.x_test) == 506


def test_vanishing_derivatives():
    # 1/x for x>1e3
    vv = 1 / np.arange(1000, 1100, step=1).reshape(10, -1)
    assert vanishing_derivatives(vv, 7, 1e-2) == 10


def test_powerset():
    assert set(powerset((1, 2))) == {(), (1,), (1, 2), (2,)}
    assert set(powerset([])) == {()}
    with pytest.raises(TypeError):
        set(powerset(1))
