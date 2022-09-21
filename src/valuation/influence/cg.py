"""
Contains

- batched conjugate gradient.
- error bound for conjugate gradient.
"""
import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from valuation.influence.types import MatrixVectorProduct
from valuation.utils import is_linear_function, is_positive_definite

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def batched_preconditioned_conjugate_gradient(
    A: Union["NDArray", Callable[["NDArray"], "NDArray"]],
    b: "NDArray",
    x0: Optional["NDArray"] = None,
    M: Optional[Union["NDArray", Callable[["NDArray"], "NDArray"]]] = None,
    rtol: float = 1e-10,
    max_iterations: int = 50,
    max_step_size: float = 10.0,
    verify_assumptions: bool = False,
    raise_exception: bool = False,
):
    """
    Implementation of a batched conjugate gradient algorithm. It uses vector matrix products for efficient calculation.
    See https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/scipy/scipy/blob/v1.8.1/scipy/sparse/linalg/_isolve/iterative.py#L282-L351 and
    https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf. On top, it constrains the maximum step size.

    :param A: A linear function f : R[k] -> R[k] representing a matrix vector product from dimension K to K or a matrix. \
        It has to be positive-definite v.T @ f(v) >= 0.
    :param b: A NDArray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param M: A function f : R[k] -> R[k] which approximates inv(A) or a matrix of shape [K, K]. The underlying matrix \
        has to be symmetric and positive definite.
    :param max_iterations: Maximum number of iterations to use in conjugate gradient. Default is 10 times K.
    :param rtol: Relative tolerance of the residual with respect to the 2-norm of b.
    :param max_step_size: Maximum step size along a gradient direction. Might be necessary for numerical stability. \
        See also max_iterations. Default is 10.0.
    :param verify_assumptions: True, iff the matrix should be checked for positive-definiteness by a stochastic rule.
    :param raise_exception: True, iff an assumption should be raised, instead of a warning only.

    :return: A NDArray of shape [K] representing the solution of Ax=b.
    """
    # wrap A into a function.
    if not callable(A):
        new_A = np.copy(A)
        A = lambda v: v @ new_A.T  # type: ignore
    if M is not None:
        if not callable(M):
            new_M = np.copy(M)
            M = lambda v: v @ new_M.T  # type: ignore

    k = A(b).shape[0]
    if A(b).size == 0:
        return b

    if verify_assumptions:
        if not is_linear_function(A, b):
            msg = "The function seems to not be linear."
            if raise_exception:
                raise Exception(msg)
            else:
                logger.warning(msg)

        if not is_positive_definite(A, b):
            msg = "The function seems to not be linear."
            if raise_exception:
                raise Exception(msg)
            else:
                logger.warning(msg)

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
        pre_alpha = r_dot_u / p_dot_mvp
        if max_step_size is not None:
            alpha = np.minimum(max_step_size, pre_alpha)

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
        msg = (
            f"Conjugate gradient could solve the equation system for {percentage_converged}% of {len(converged)} random"
            " chosen vectors. Please check condition number and eigenvalues."
        )
        if raise_exception:
            raise Exception(msg)
        else:
            logger.warning(msg)

    return x, iteration


def conjugate_gradient_condition_number_based_error_bound(
    A: "NDArray", n: int, x0: "NDArray", xt: "NDArray"
) -> float:
    """
    Error bound for conjugate gradient based on the condition number of the weight matrix A. Used for testing purposes.
    See also https://math.stackexchange.com/questions/382958/error-for-conjugate-gradient-method. Explicit of the weight
    matrix is required.
    :param A: Weight matrix of the matrix to be inverted.
    :param n: Maximum number for executed iterations X in conjugate gradient.
    :param x0: Initialization solution x0 of conjugate gradient.
    :param xt: Final solution xt of conjugate gradient after X iterations.
    :returns: Upper bound for ||x0 - xt||_A.
    """
    eigvals = np.linalg.eigvals(A)
    eigvals = np.sort(eigvals)
    eig_val_max = np.max(eigvals)
    eig_val_min = np.min(eigvals)
    kappa = np.abs(eig_val_max / eig_val_min)
    norm_A = lambda v: np.sqrt(np.einsum("ia,ab,ib->i", v, A, v))
    error_init: float = norm_A(xt - x0)

    sqrt_kappa = np.sqrt(kappa)
    div = (sqrt_kappa + 1) / (sqrt_kappa - 1)
    div_n = div**n
    return (2 * error_init) / (div_n + 1 / div_n)  # type: ignore


def hvp_to_inv_diag_conditioner(
    hvp: MatrixVectorProduct, d: int
) -> MatrixVectorProduct:
    """
    This method uses the hvp function to construct a simple pre-conditioner 1/diag(H). It does so while requiring
    only O(d) space in RAM for construction and later execution.
    :param hvp: The callable calculating the Hessian vector product Hv.
    :param d: The number of dimensions of the hvp callable.
    :returns: A MatrixVectorProduct for the conditioner.
    """
    diags = np.empty(d)

    for i in range(d):
        inp = np.zeros(d)
        inp[i] = 1
        diags[i] = hvp(np.reshape(inp, [1, -1]))[0, i]

    def _inv_diag_conditioner(v: "NDArray"):
        return v / diags

    return _inv_diag_conditioner
