from types import LambdaType
from typing import Callable, Union

import numpy as np
from opt_einsum import contract
from scipy.sparse.linalg._isolve.iterative import _get_atol


def conjugate_gradient(
    A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    b: np.ndarray,
    x0: np.ndarray = None,
    tol: float = 1e-5,
    max_iters: int = None,
):
    """
    Implementation of a batched conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm. See also
    https://github.com/scipy/scipy/blob/v1.8.1/scipy/sparse/linalg/_isolve/iterative.py#L282-L351 and


    :param A: A function f : R[n] -> R[n] representing a matrix vector product from dimension K to K or a matrix
    of shape [K, K]
    :param b: A np.ndarray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param max_iters: The maximum number of iterations to use in conjugate gradient.
    :param tol: Damping for vector products controls when the iteration is stopped.
    :return: A np.ndarray of shape [K] representing the solution of Ax=b.
    """

    # wrap A into a function.
    A_fn = A if isinstance(A, LambdaType) else lambda v: v @ A.T
    if A_fn(b).size == 0:
        return b

    if b.ndim == 1:
        b = b.reshape([1, -1])

    if max_iters is None:
        max_iters = 10 * b.shape[-1]

    # start with residual
    x = np.zeros_like(b) if x0 is None else x0
    r = np.copy(b)
    p = np.copy(b)

    if x.ndim == 1:
        x = x.reshape([1, -1])

    iteration = 0
    r_dot_r = contract("ia,ia->i", r, r)
    batch_dim = b.shape[0]
    converged = np.zeros(batch_dim, dtype=bool)
    atol = np.linalg.norm(b, axis=1) * tol

    while iteration < max_iters:

        if np.any(np.isnan(p)):
            print("Jagger")

        # remaining fields
        iteration += 1
        not_yet_converged_indices = np.argwhere(np.logical_not(converged))[:, 0]
        mvp = A_fn(p)[not_yet_converged_indices]
        p_dot_mvp = contract("ia,ia->i", p[not_yet_converged_indices], mvp)
        alpha = r_dot_r[not_yet_converged_indices] / p_dot_mvp

        # update x and r
        reshaped_alpha = alpha.reshape([-1, 1])
        x[not_yet_converged_indices] += reshaped_alpha * p[not_yet_converged_indices]
        r[not_yet_converged_indices] -= reshaped_alpha * mvp

        # calculate next conjugate gradient
        new_r_dot_r = contract(
            "ia,ia->i", r[not_yet_converged_indices], r[not_yet_converged_indices]
        )

        if tol is not None:
            residual = np.linalg.norm(
                A_fn(x)[not_yet_converged_indices] - b[not_yet_converged_indices],
                axis=1,
            )
            converged[not_yet_converged_indices] = (
                residual <= atol[not_yet_converged_indices]
            )

        if np.all(converged):
            break

        beta = new_r_dot_r / r_dot_r[not_yet_converged_indices]
        p[not_yet_converged_indices] = (
            beta.reshape([-1, 1]) * p[not_yet_converged_indices]
            + r[not_yet_converged_indices]
        )
        r_dot_r[not_yet_converged_indices] = new_r_dot_r

    return x, iteration
