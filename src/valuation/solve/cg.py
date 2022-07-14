from types import LambdaType
from typing import Callable, Union

import numpy as np

from valuation.utils import (
    mcmc_is_linear_function,
    mcmc_is_linear_function_positive_definite,
)
from valuation.utils.logging import raise_or_log


def conjugate_gradient(
    A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    b: np.ndarray,
    x0: np.ndarray = None,
    M: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-10,
    max_iterations: int = None,
    verify_assumptions: bool = False,
    raise_exception: bool = False,
    max_step_size: float = 10.0,
):
    """
    Implementation of a batched conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/scipy/scipy/blob/v1.8.1/scipy/sparse/linalg/_isolve/iterative.py#L282-L351 and
    https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf.
    :param A: A linear function f : R[k] -> R[k] representing a matrix vector product from dimension K to K or a matrix.
    It has to be positive-definite v.T @ f(v) >= 0.
    :param b: A np.ndarray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param M: A function f : R[k] -> R[k] which approximates inv(A) or a matrix of shape [K, K]. The underlying matrix
    has to be symmetric and positive definite.
    :param max_iterations: The maximum number of iterations to use in conjugate gradient.
    :param rtol: Relative tolerance of the residual with respect to the 2-norm of b.
    :param verify_assumptions: True, iff the matrix should be checked for positive-definiteness by a stochastic rule.
    :param raise_exception: True, iff an assumption should be raised, instead of a warning only.
    :return: A np.ndarray of shape [K] representing the solution of Ax=b.
    """

    # wrap A into a function.
    new_A = np.copy(A)
    A = A if isinstance(A, LambdaType) else lambda v: v @ new_A.T
    if M is not None:
        if isinstance(M, LambdaType):
            M = M
        else:
            new_M = np.copy(M)
            M = lambda v: v @ new_M.T

    k = A(b).shape[0]
    if A(b).size == 0:
        return b

    if verify_assumptions:
        if not mcmc_is_linear_function(A, b):
            raise_or_log(
                "The function seems to not be linear.", raise_exception=raise_exception
            )

        if not mcmc_is_linear_function_positive_definite(A, b):
            raise_or_log(
                "The function seems to not be linear.", raise_exception=raise_exception
            )

    if b.ndim == 1:
        b = b.reshape([1, -1])

    if max_iterations is None:
        max_iterations = 10 * k

    # start with residual
    if x0 is not None:
        x = np.copy(x0)
    elif M is not None:
        x = M(b)
    else:
        x = np.copy(b)

    r = b - A(x)
    u = np.copy(r)

    if M is not None:
        u = M(u)

    p = np.copy(b)

    if x.ndim == 1:
        x = x.reshape([1, -1])

    iteration = 0
    batch_dim = b.shape[0]
    converged = np.zeros(batch_dim, dtype=bool)
    atol = np.linalg.norm(b, axis=1) * rtol

    while iteration < max_iterations:

        # remaining fields
        iteration += 1
        not_yet_converged_indices = np.argwhere(np.logical_not(converged))[:, 0]
        mvp = A(p)[not_yet_converged_indices]
        p_dot_mvp = np.einsum("ia,ia->i", p[not_yet_converged_indices], mvp)
        r_dot_u = np.einsum(
            "ia,ia->i", r[not_yet_converged_indices], u[not_yet_converged_indices]
        )
        alpha = np.minimum(max_step_size, r_dot_u / p_dot_mvp)

        # update x and r
        reshaped_alpha = alpha.reshape([-1, 1])
        x[not_yet_converged_indices] += reshaped_alpha * p[not_yet_converged_indices]
        r[not_yet_converged_indices] -= reshaped_alpha * mvp

        # calculate next conjugate gradient
        new_u = r
        if M is not None:
            new_u = M(new_u)

        new_u = new_u[not_yet_converged_indices]
        new_r_dot_u = np.einsum("ia,ia->i", r[not_yet_converged_indices], new_u)

        if rtol is not None:
            residual = np.linalg.norm(
                A(x)[not_yet_converged_indices] - b[not_yet_converged_indices],
                axis=1,
            )
            converged[not_yet_converged_indices] = (
                residual <= atol[not_yet_converged_indices]
            )

        if np.all(converged):
            break

        beta = new_r_dot_u / r_dot_u
        p[not_yet_converged_indices] = (
            beta.reshape([-1, 1]) * p[not_yet_converged_indices] + new_u
        )
        u[not_yet_converged_indices] = new_u

    if not np.all(converged):
        percentage_converged = int(converged.sum() / len(converged)) * 100
        raise_or_log(
            f"Conjugate gradient could solve the equation system for {percentage_converged}% of {len(converged)} random"
            f" chosen vectors. Please check condition number and eigenvalues of",
            raise_exception=raise_exception,
        )

    return x, iteration


def conjugate_gradient_error_bound(
    A: np.ndarray, n: int, x0: np.ndarray, xt: np.ndarray
) -> float:
    """
    https://math.stackexchange.com/questions/382958/error-for-conjugate-gradient-method
    """
    eigvals = np.linalg.eigvals(A)
    eigvals = np.sort(eigvals)
    eig_val_max = np.max(eigvals)
    eig_val_min = np.min(eigvals)
    kappa = np.abs(eig_val_max / eig_val_min)
    norm_A = lambda v: np.sqrt(np.einsum("ia,ab,ib->i", v, A, v))
    error_init = norm_A(xt - x0)

    sqrt_kappa = np.sqrt(kappa)
    div = (sqrt_kappa + 1) / (sqrt_kappa - 1)
    div_n = div**n
    return (2 * error_init) / (div_n + 1 / div_n)
