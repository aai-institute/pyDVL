import pytest
import torch

from pydvl.influence.torch.pre_conditioner import (
    JacobiPreConditioner,
    NystroemPreConditioner,
)


def high_cond_diagonal_dominant_matrix(size, high_value=1e5, low_value=1e-2):
    """Generates a diagonal dominant matrix with a high condition number."""
    A = torch.randn(size, size) * low_value  # Small off-diagonal elements
    for i in range(size):
        A[i, i] = high_value if i % 2 == 0 else low_value

    return A.T @ A


def approx_low_rank_matrix(size, rank):
    """Generates an approximately low-rank matrix."""
    U = torch.randn(size, rank)
    return U @ U.T + 1e-1 * torch.eye(size)


@pytest.fixture
def high_cond_mat():
    size = 100  # Example size
    return high_cond_diagonal_dominant_matrix(size)


@pytest.fixture
def low_rank_mat():
    size = 1000  # Example size
    rank = 50
    return approx_low_rank_matrix(size, rank)


@pytest.mark.parametrize("num_samples_estimator", [1, 3, 5])
def test_jacobi_preconditioner_condition_number(high_cond_mat, num_samples_estimator):
    preconditioner = JacobiPreConditioner(num_samples_estimator=num_samples_estimator)
    size = high_cond_mat.shape[0]
    regularization = 0.1

    # Original matrix and its condition number
    A = high_cond_mat
    original_cond_number = torch.linalg.cond(A + regularization * torch.eye(size))

    preconditioner.fit(
        lambda x: A @ x, size, high_cond_mat.dtype, high_cond_mat.device, regularization
    )
    assert preconditioner.is_fitted

    preconditioned_A = preconditioner.solve(A + regularization * torch.eye(size))
    preconditioned_cond_number = torch.linalg.cond(preconditioned_A)

    # Assert that the condition number has decreased
    assert preconditioned_cond_number < original_cond_number * 10 ** (
        -0.5 * (num_samples_estimator)
    )


def test_nystroem_preconditioner_condition_number(low_rank_mat):
    preconditioner = NystroemPreConditioner(60)
    size = low_rank_mat.shape[0]
    regularization = 1e-2

    # Original matrix and its condition number
    original_cond_number = torch.linalg.cond(
        low_rank_mat + regularization * torch.eye(size)
    )

    preconditioner.fit(
        lambda x: low_rank_mat @ x,
        low_rank_mat.shape[0],
        low_rank_mat.dtype,
        low_rank_mat.device,
        regularization,
    )
    assert preconditioner.is_fitted

    preconditioned_A = preconditioner.solve(
        low_rank_mat + regularization * torch.eye(size)
    )
    preconditioned_cond_number = torch.linalg.cond(preconditioned_A)

    # Assert that the condition number has decreased
    assert preconditioned_cond_number < original_cond_number * 1e-1
