"""
Contains

- batched conjugate gradient.
- error bound for conjugate gradient.
"""
import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import cg

from ..utils import maybe_progress

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["conjugate_gradient", "batched_preconditioned_conjugate_gradient"]

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"


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
        y_cg, _ = cg(A, y, atol="legacy")
        batch_cg.append(y_cg)
    return np.asarray(batch_cg)
