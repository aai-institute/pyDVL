import numpy as np
import pytest
from valuation.utils.algorithms import conjugate_gradient
from opt_einsum import contract

class AlgorithmTestSettings:
    R_TOL = 0.2
    FAILED_TOL = 0.1
    CG_DAMPING = 1e-10
    CG_BATCH_SIZE = 100
    CG_MAX_DIM = 100
    NP_RANDOM_SEED = 42


np.random.seed(AlgorithmTestSettings.NP_RANDOM_SEED)
conjugate_gradient_examples = [
    (
        np.random.random([k, k]),
        np.random.normal(size=[AlgorithmTestSettings.CG_BATCH_SIZE, k]),
        fn,
    )
    for k in range(2, AlgorithmTestSettings.CG_MAX_DIM)
    for fn in [True, False]
]


@pytest.mark.parametrize("A,b,wrap_in_function", conjugate_gradient_examples)
def test_conjugate_gradients_direct(
    A: np.ndarray, b: np.ndarray, wrap_in_function: bool
):

    A = A @ A.T
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(
        (lambda x: x @ A.T) if wrap_in_function else A, b, x0=x0
    )
    assert np.all(np.logical_not(np.isnan(xn)))

    inv_A = np.linalg.inv(A)
    xt = b @ inv_A.T
    norm_A = lambda v: np.sqrt(contract('ia,ab,ib->i', v, A, v))
    error = norm_A(xt - xn)
    error_upper_bound = conjugate_gradient_error_bound(A, xt, x0, n)
    failed = error > (1 + AlgorithmTestSettings.R_TOL) * error_upper_bound
    num_failed_percentage = np.sum(failed) / len(failed)
    assert num_failed_percentage < AlgorithmTestSettings.FAILED_TOL


def conjugate_gradient_error_bound(A: np.ndarray, xt: np.ndarray, x0: np.ndarray, n: int):
    """
    https://math.stackexchange.com/questions/382958/error-for-conjugate-gradient-method
    """
    norm_A = lambda v: np.sqrt(contract('ia,ab,ib->i', v, A, v))
    error_init = norm_A(xt - x0)
    kappa = np.linalg.cond(A)
    sqrt_kappa = np.sqrt(kappa)
    div = (sqrt_kappa + 1) / (sqrt_kappa - 1)
    div_n = div ** n
    return (2 * error_init) / (div_n + 1 / div_n)