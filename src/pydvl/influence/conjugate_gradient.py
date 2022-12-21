"""
Contains

- batched conjugate gradient.
- error bound for conjugate gradient.
"""
import logging
import warnings
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
from scipy.sparse.linalg import cg

from ..utils import maybe_progress
from .types import MatrixVectorProduct

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["conjugate_gradient", "batched_preconditioned_conjugate_gradient"]

logger = logging.getLogger(__name__)


def conjugate_gradient(
    A: "NDArray[np.float_]", batch_y: "NDArray[np.float_]", progress: bool = False
) -> "NDArray[np.float_]":
    """
    Given a matrix and a batch of vectors, it uses conjugate gradient to calculate the solution
    to Ax = y for each y in batch_y.

    :param A: a real, symmetric and positive-definite matrix of shape [NxN]
    :param batch_y: a matrix of shape [NxP], with P the size of the batch.
    :param progress: True, iff progress shall be printed.

    :return: A NDArray of shape [NxP] representing x, the solution of Ax=b.
    """
    batch_cg = []
    for y in maybe_progress(batch_y, progress, desc="Conjugate gradient"):
        y_cg, _ = cg(A, y)
        batch_cg.append(y_cg)
    return np.asarray(batch_cg)


def batched_preconditioned_conjugate_gradient(
    A: Union["NDArray", Callable[["NDArray"], "NDArray"]],
    b: "NDArray",
    x0: Optional["NDArray"] = None,
    rtol: float = 1e-3,
    max_iterations: int = 100,
    max_step_size: Optional[float] = None,
) -> Tuple["NDArray", int]:
    """
    Implementation of a batched conjugate gradient algorithm. It uses vector matrix products for efficient calculation.
    On top of that, it constrains the maximum step size.

    See [1]_ for more details on the algorithm.

    See also [2]_ and [3]_.

    .. warning::

        This function is experimental and unstable. Prefer using inversion_method='cg'

    :param A: A linear function f : R[k] -> R[k] representing a matrix vector product from dimension K to K or a matrix. \
        It has to be positive-definite v.T @ f(v) >= 0.
    :param b: A NDArray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param max_iterations: Maximum number of iterations to use in conjugate gradient. Default is 10 times K.
    :param rtol: Relative tolerance of the residual with respect to the 2-norm of b.
    :param max_step_size: Maximum step size along a gradient direction. Might be necessary for numerical stability. \
        See also max_iterations. Default is 10.0.
    :param verify_assumptions: True, iff the matrix should be checked for positive-definiteness by a stochastic rule.

    :return: A NDArray of shape [K] representing the solution of Ax=b.

    .. note::
        .. [1] `Conjugate Gradient Method - Wikipedia <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.
        .. [2] `SciPy's implementation of Conjugate Gradient <https://github.com/scipy/scipy/blob/v1.8.1/scipy/sparse/linalg/_isolve/iterative.py#L282-L351>`_.
        .. [3] `Prof. Mert Pilanci., "Conjugate Gradient Method", Stanford University, 2022 <https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf>`_.
    """
    warnings.warn(
        "This function is experimental and unstable. Prefer using inversion_method='cg'",
        UserWarning,
    )
    # wrap A into a function.
    if not callable(A):
        new_A = np.copy(A)
        A = lambda v: v @ new_A.T  # type: ignore
    M = hvp_to_inv_diag_conditioner(A, d=b.shape[1])

    k = A(b).shape[0]
    if A(b).size == 0:
        return b, 0

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
        alpha = r_dot_u / p_dot_mvp
        if max_step_size is not None:
            alpha = np.minimum(max_step_size, alpha)

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
            f"Converged vectors are only {percentage_converged}%. "
            "Please increase max_iterations, decrease max_step_size "
            "and make sure that A is positive definite"
            " (e.g. through regularization)."
        )
        warnings.warn(msg, RuntimeWarning)
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
