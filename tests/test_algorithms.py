import numpy as np
import pytest

from valuation.utils.algorithms import conjugate_gradient


diff_tol = 1e-3

np.random.seed(42)
conjugate_gradient_examples = [
    (np.random.normal(size=(3, 3)), np.random.normal(size=[3]))
    for k in range(10)
    # TODO add test for batch.
]


@pytest.mark.parametrize("A,b", conjugate_gradient_examples)
def test_conjugate_gradients_direct(A: np.ndarray, b: np.ndarray):

    A = A @ A.T
    x = conjugate_gradient(A, b, max_iter=20, damping=1e-3)
    diff = np.linalg.norm(x @ A.T - b)
    assert diff < diff_tol


@pytest.mark.parametrize("A,b", conjugate_gradient_examples)
def test_conjugate_gradients_indirect(A: np.ndarray, b: np.ndarray):

    A = A @ A.T
    hvp = lambda x: x @ A.T
    x = conjugate_gradient(hvp, b, max_iter=20, damping=1e-3)
    diff = np.linalg.norm(x @ A.T - b)
    assert diff < diff_tol
