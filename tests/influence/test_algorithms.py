import numpy as np
import pytest
from opt_einsum import contract

from valuation.utils.algorithms import conjugate_gradient


class AlgorithmTestSettings:
    R_TOL = 1.0
    FAILED_TOL = 0.1
    CG_DAMPING = 1e-10
    CG_BATCH_SIZE = 100
    CG_MAX_DIM = 20
    NP_RANDOM_SEED = 42


np.random.seed(AlgorithmTestSettings.NP_RANDOM_SEED)

conjugate_gradient_examples = list()
ids = list()

# non-singular tests
conjugate_gradient_examples += [
    (
        (lambda A: A @ A.T)(np.random.random([k, k])),
        np.random.normal(size=[AlgorithmTestSettings.CG_BATCH_SIZE, k]),
        fn,
    )
    for k in range(2, AlgorithmTestSettings.CG_MAX_DIM)
    for fn in [True, False]
]
ids += [
    f"Testcase with dimension {tup[0].shape[0]}." for tup in conjugate_gradient_examples
]

# singular tests
conjugate_gradient_examples += [
    (
        np.asarray(
            [
                [3, 8, 1],
                [-4, 1, 1],
                [-4, 1, 1],
            ],
            dtype=float,
        ),
        np.asarray([[1, 2, 3]], dtype=float),
        False,
    )
]
ids += ["Singular testcase."]


@pytest.mark.parametrize("A,b,wrap_in_function", conjugate_gradient_examples, ids=ids)
def test_conjugate_gradients_all(A: np.ndarray, b: np.ndarray, wrap_in_function: bool):
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient((lambda x: x @ A.T) if wrap_in_function else A, b, x0=x0)
    assert np.all(np.logical_not(np.isnan(xn)))

    inv_A = np.linalg.pinv(A)
    xt = b @ inv_A.T
    norm_A = lambda v: np.sqrt(contract("ia,ab,ib->i", v, A, v))
    error = norm_A(xt - xn)
    error_upper_bound = conjugate_gradient_error_bound(A, xt, x0, n)
    failed = error > (1 + AlgorithmTestSettings.R_TOL) * error_upper_bound
    num_failed_percentage = np.sum(failed) / len(failed)
    assert num_failed_percentage < AlgorithmTestSettings.FAILED_TOL


@pytest.mark.parametrize("A,b,wrap_in_function", conjugate_gradient_examples, ids=ids)
def test_preconditioned_conjugate_gradients_all(
    A: np.ndarray, b: np.ndarray, wrap_in_function: bool
):
    M = np.diag(1 / np.diag(A))
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(
        (lambda x: x @ A.T) if wrap_in_function else A, b, M=M, x0=x0
    )
    assert np.all(np.logical_not(np.isnan(xn)))

    inv_A = np.linalg.pinv(A)
    xt = b @ inv_A.T
    norm_A = lambda v: np.sqrt(contract("ia,ab,ib->i", v, A, v))
    error = norm_A(xt - xn)
    error_upper_bound = conjugate_gradient_error_bound(A, xt, x0, n)
    failed = error > (1 + AlgorithmTestSettings.R_TOL) * error_upper_bound
    num_failed_percentage = np.sum(failed) / len(failed)
    assert num_failed_percentage < AlgorithmTestSettings.FAILED_TOL


def conjugate_gradient_error_bound(
    A: np.ndarray, xt: np.ndarray, x0: np.ndarray, n: int
):
    """
    https://math.stackexchange.com/questions/382958/error-for-conjugate-gradient-method
    """
    norm_A = lambda v: np.sqrt(contract("ia,ab,ib->i", v, A, v))
    error_init = norm_A(xt - x0)
    kappa = np.linalg.cond(A)
    sqrt_kappa = np.sqrt(kappa)
    div = (sqrt_kappa + 1) / (sqrt_kappa - 1)
    div_n = div**n
    return (2 * error_init) / (div_n + 1 / div_n)
