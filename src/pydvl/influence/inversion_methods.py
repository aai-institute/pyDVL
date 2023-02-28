"""
Contains

- batched conjugate gradient.
- error bound for conjugate gradient.
"""
import logging
from enum import Enum
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg

from ..utils import maybe_progress
from .frameworks import TensorType

__all__ = ["matrix_inversion_algorithm", "conjugate_gradient"]

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"


def invert_matrix(
    inversion_method: InversionMethod,
    mvp: Callable[[TensorType], NDArray[np.float_]],
    mvp_dimensions: Tuple[int, int],
    b: NDArray[np.float_],
    progress: bool = False,
) -> NDArray[np.float_]:
    """
    Finds $x$ such that $Ax = b$, where $A$ is a matrix of size ``mvp_dimensions``),
    and $b$ a vector.
    Instead of passing the matrix A directly, the method takes its mvp, i.e. a
    (callable) matrix vector product that returns the product Ax for any vector
    x. This is done to avoid storing a (potentially very large) matrix in memory.

    :param inversion_method:
    :param mvp: matrix vector product, a callable that, given any array $x$,
        returns the result of $Ax$.
    :param mvp_dimensions: dimensions of matrix $A$
    :param b:
    :param progress: If True, display progress bars.

    :return: An array that solves the inverse problem,
        i.e. it returns $x$ such that $Ax = b$
    """
    if inversion_method == InversionMethod.Direct:
        return (np.linalg.inv(mvp(np.eye(mvp_dimensions[1]))) @ b.T).T
    elif inversion_method == InversionMethod.Cg:
        return conjugate_gradient(
            LinearOperator(mvp_dimensions, matvec=mvp), b, progress
        )
    else:
        raise ValueError(f"Unknown inversion method: {inversion_method}")


def conjugate_gradient(
    A: LinearOperator, batch_y: NDArray[np.float_], progress: bool = False
) -> NDArray[np.float_]:
    """
    Given a matrix and a batch of vectors, it uses conjugate gradient to calculate the solution
    to $Ax = y$ for each $y$ in ``batch_y``.

    :param A: a real, symmetric and positive-definite matrix of shape [NxN]
    :param batch_y: a matrix of shape [PxN], with P the size of the batch.
    :param progress: True, iff progress shall be printed.

    :return: A matrix of shape [PxN] with each line being a solution of $Ax=b$.
    """
    batch_cg = []
    for y in maybe_progress(batch_y, progress, desc="Conjugate gradient"):
        y_cg, _ = cg(A, y, atol="legacy")
        batch_cg.append(y_cg)
    return np.asarray(batch_cg)
