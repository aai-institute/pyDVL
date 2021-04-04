import numpy as np

from valuation.utils import vanishing_derivatives


def test_dataset_len(dataset):
    assert len(dataset) == len(dataset.x_train) == 404
    assert len(dataset.x_train) + len(dataset.x_test) == 506


def test_vanishing_derivatives():
    # 1/x for x>1e3
    vv = 1 / np.arange(1000, 1100, step=1).reshape(10, -1)
    assert vanishing_derivatives(vv, 7, 1e-2) == 10
