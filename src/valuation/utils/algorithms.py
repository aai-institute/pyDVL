from types import LambdaType
from typing import Callable, Union

import numpy as np


def conjugate_gradient(
    A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    b: np.ndarray,
    max_iter: int = 10,
    damping: float = 1e-10,
):
    """
    Implementation of conjugate gradient algorithm. It uses vector matrix products for efficient calculation. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details of the algorithm.

    :param A: A function f : R[n] -> R[n] representing a matrix vector product from dimension K to K or a matrix
    of shape [K, K]
    :param b: A np.ndarray of shape [K] representing the targeted result of the matrix multiplication Ax.
    :param max_iter: The maximum number of iterations to use in conjugate gradient.
    :param damping: Damping for vector products controls when the iteration is stopped.
    :return: A np.ndarray of shape [K] representing the solution of Ax=b.
    """

    is_lambda = isinstance(A, LambdaType)
    if not is_lambda:
        A_fn = lambda v: v @ A.T
    else:
        A_fn = A

    # start with residual
    x = np.zeros_like(b)
    r = np.copy(b)
    p = np.copy(b)

    iteration = 0
    r_dot_r = np.dot(r, r)

    while iteration < max_iter:

        # remaining fields
        matrix_vector_product = A_fn(p)
        alpha = r_dot_r / np.dot(p, matrix_vector_product)

        # update x and r
        x += alpha * p
        r -= alpha * matrix_vector_product

        new_r_dot_r = np.dot(r, r)
        beta = new_r_dot_r / r_dot_r
        p = beta * p + r
        r_dot_r = new_r_dot_r
        if r_dot_r <= damping:
            break

        # calculate next conjugate gradient
        iteration += 1

    return x
