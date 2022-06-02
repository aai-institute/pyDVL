import itertools
from typing import Tuple

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


@pytest.mark.parametrize(
    "problem_dimension,batch_size",
    list(itertools.product(list(range(2, 10)), [16, 32])),
    indirect=True,
)
def test_conjugate_gradients_mvp(linear_equation_system: Tuple[np.ndarray, np.ndarray]):
    A, b = linear_equation_system
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(A, b, x0=x0)
    check_solution(A, b, n, x0, xn)


@pytest.mark.parametrize(
    "problem_dimension,batch_size",
    list(itertools.product(list(range(2, 10)), [16, 32])),
    indirect=True,
)
def test_conjugate_gradients_fn(linear_equation_system: Tuple[np.ndarray, np.ndarray]):
    A, b = linear_equation_system
    new_A = np.copy(A)
    A = lambda v: v @ new_A.T
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(A, b, x0=x0)
    check_solution(new_A, b, n, x0, xn)


@pytest.mark.parametrize(
    "problem_dimension,batch_size",
    list(itertools.product(list(range(2, 10)), [16, 32])),
    indirect=True,
)
def test_conjugate_gradients_mvp_preconditioned(
    linear_equation_system: Tuple[np.ndarray, np.ndarray]
):
    A, b = linear_equation_system
    M = np.diag(1 / np.diag(A))
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(A, b, M=M, x0=x0)
    check_solution(A, b, n, x0, xn)


def check_solution(A, b, n, x0, xn):
    """
    This method used standard inversion techniques to verify the solution of the problem.
    """
    assert np.all(np.logical_not(np.isnan(xn)))
    inv_A = np.linalg.pinv(A)
    xt = b @ inv_A.T
    norm_A = lambda v: np.sqrt(contract("ia,ab,ib->i", v, A, v))
    error = norm_A(xt - xn)
    error_upper_bound = conjugate_gradient_error_bound_first(A, n, x0, xt)
    failed = error > (1 + AlgorithmTestSettings.R_TOL) * error_upper_bound
    num_failed_percentage = np.sum(failed) / len(failed)
    assert num_failed_percentage < AlgorithmTestSettings.FAILED_TOL


def conjugate_gradient_error_bound_first(
    A: np.ndarray, n: int, x0: np.ndarray, xt: np.ndarray
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
