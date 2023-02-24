"""
Contains

- batched conjugate gradient.
- error bound for conjugate gradient.
"""
import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ..utils import maybe_progress

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["matrix_inversion_algorithm", "conjugate_gradient"]

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"


MatrixVectorProduct = Callable[["NDArray"], "NDArray"]


def matrix_inversion_algorithm(
    inversion_method: InversionMethod,
    mvp: MatrixVectorProduct,
    mvp_dimensions: Tuple[int, int],
    progress: bool = False,
) -> Callable[["NDArray"], "NDArray"]:
    """
    It returns a callable that solves the problem of finding x such
    that Ax = b, where A is a matrix (of size mvp_dimensions), and b a vector.
    More precisely, given mvp, a (callable) matrix vector product that
    returns the product Ax for any vector x, it returns another callable that takes
    any vector b and solves the inverse problem.

    :param inversion_method:
    :param mvp: matrix vector product, a callable that, given any array x,
        returns the result of Ax.
    :param mvp_dimensions: dimensions of matrix A
    :param progress: If True, display progress bars.

    :return: A callable that, given any array b, solves the inverse problem,
        i.e. it finds x such that Ax = b
    """
    inversion_algorithm_registry = {
        InversionMethod.Direct: lambda x: np.linalg.solve(mvp(np.eye(mvp_dimensions[0])), x.T).T,  # type: ignore
        InversionMethod.Cg: lambda x: conjugate_gradient(LinearOperator(mvp_dimensions, matvec=mvp), x, progress),  # type: ignore
    }
    return inversion_algorithm_registry[inversion_method]


def conjugate_gradient(
    A: "NDArray[np.float_]", batch_y: "NDArray[np.float_]", progress: bool = False
) -> "NDArray[np.float_]":
    """
    Given a matrix and a batch of vectors, it uses conjugate gradient to calculate the solution
    to Ax = y for each y in batch_y.

    :param A: a real, symmetric and positive-definite matrix of shape [NxN]
    :param batch_y: a matrix of shape [PxN], with P the size of the batch.
    :param progress: True, iff progress shall be printed.

    :return: A NDArray of shape [PxN] with each line being a solution of Ax=b.
    """
    batch_cg = []
    for y in maybe_progress(batch_y, progress, desc="Conjugate gradient"):
        y_cg, _ = cg(A, y, atol="legacy")
        batch_cg.append(y_cg)
    return np.asarray(batch_cg)
