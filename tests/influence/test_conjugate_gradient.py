import itertools
from typing import List, Tuple

import numpy as np
import pytest
from opt_einsum import contract

from valuation.utils.cg import conjugate_gradient, conjugate_gradient_error_bound


class AlgorithmTestSettings:
    L2_TOL = 0.01
    FAILED_TOL = 0.1
    CG_DAMPING = 1e-10

    CG_TEST_CONDITION_NUMBERS: List[int] = [10]
    CG_TEST_BATCH_SIZES: List[int] = [16, 32]
    CG_TEST_DIMENSIONS: List[int] = list(np.arange(2, 100, 5))


test_cases = list(
    itertools.product(
        AlgorithmTestSettings.CG_TEST_DIMENSIONS,
        AlgorithmTestSettings.CG_TEST_BATCH_SIZES,
        AlgorithmTestSettings.CG_TEST_CONDITION_NUMBERS,
    )
)
lmb_test_case_to_str = lambda packed_i_test_case: (
    lambda i, test_case: f"Problem #{i} of dimension {test_case[0]} with batch size {test_case[1]} and condition number"
    f" {test_case[2]}"
)(*packed_i_test_case)
test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


@pytest.mark.parametrize(
    "problem_dimension,batch_size,condition_number",
    test_cases,
    ids=test_case_ids,
    indirect=True,
)
def test_conjugate_gradients_mvp(linear_equation_system: Tuple[np.ndarray, np.ndarray]):
    A, b = linear_equation_system
    x0 = np.zeros_like(b)
    xn, n = conjugate_gradient(A, b, x0=x0)
    check_solution(A, b, n, x0, xn)


@pytest.mark.parametrize(
    "problem_dimension,batch_size,condition_number",
    test_cases,
    ids=test_case_ids,
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
    "problem_dimension,batch_size,condition_number",
    test_cases,
    ids=test_case_ids,
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
    Uses standard inversion techniques to verify the solution of the problem.
    """
    assert np.all(np.logical_not(np.isnan(xn)))
    inv_A = np.linalg.inv(A)
    xt = b @ inv_A.T
    bound = conjugate_gradient_error_bound(A, n, x0, xt)
    norm_A = lambda v: np.sqrt(contract("ia,ab,ib->i", v, A, v))
    error = norm_A(xt - xn)
    failed = error > AlgorithmTestSettings.L2_TOL
    num_failed_percentage = np.sum(failed) / len(failed)
    assert num_failed_percentage < AlgorithmTestSettings.FAILED_TOL
