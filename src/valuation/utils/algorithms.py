from types import LambdaType
from typing import Callable, Union

import numpy as np
import scipy
from opt_einsum import contract


def conjugate_gradient(
    A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    b: np.ndarray,
    x0: np.ndarray = None,
    M: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-5,
    max_iterations: int = None,
):
    """
    Implementation of a batched conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/scipy/scipy/blob/v1.8.1/scipy/sparse/linalg/_isolve/iterative.py#L282-L351 and
    https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf.

    :param A: A function f : R[k] -> R[k] representing a matrix vector product from dimension K to K or a matrix
    of shape [K, K]. The underlying matrix has to be symmetric and positive definite.
    :param b: A np.ndarray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param M: A function f : R[k] -> R[k] which approximates inv(A) or a matrix of shape [K, K]. The underlying matrix
    has to be symmetric and positive definite.
    :param max_iterations: The maximum number of iterations to use in conjugate gradient.
    :param tol: Damping for vector products controls when the iteration is stopped.
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

    if b.ndim == 1:
        b = b.reshape([1, -1])

    if max_iterations is None:
        max_iterations = k

    # start with residual
    x = np.zeros_like(b) if x0 is None else x0
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
    atol = np.linalg.norm(b, axis=1) * tol

    while iteration < max_iterations:

        # remaining fields
        iteration += 1
        not_yet_converged_indices = np.argwhere(np.logical_not(converged))[:, 0]
        mvp = A(p)[not_yet_converged_indices]
        p_dot_mvp = contract("ia,ia->i", p[not_yet_converged_indices], mvp)
        r_dot_u = contract(
            "ia,ia->i", r[not_yet_converged_indices], u[not_yet_converged_indices]
        )
        alpha = r_dot_u / p_dot_mvp

        # update x and r
        reshaped_alpha = alpha.reshape([-1, 1])
        x[not_yet_converged_indices] += reshaped_alpha * p[not_yet_converged_indices]
        r[not_yet_converged_indices] -= reshaped_alpha * mvp

        # calculate next conjugate gradient
        new_u = r
        if M is not None:
            new_u = M(new_u)

        new_u = new_u[not_yet_converged_indices]
        new_r_dot_u = contract("ia,ia->i", r[not_yet_converged_indices], new_u)

        if tol is not None:
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

    return x, iteration
