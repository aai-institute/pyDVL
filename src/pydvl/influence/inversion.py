"""
Contains methods to invert the hessian vector product. 
"""
import logging
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg

from ..utils import maybe_progress
from .frameworks import (
    ModelType,
    TensorType,
    TwiceDifferentiable,
    as_tensor,
    identity_tensor,
    mvp,
    solve_batch_cg,
    solve_linear,
)

__all__ = ["solve_hvp"]

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"


def solve_hvp(
    inversion_method: InversionMethod,
    model: TwiceDifferentiable[TensorType, ModelType],
    x: TensorType,
    y: TensorType,
    b: TensorType,
    lam: float = 0,
    progress: bool = False,
) -> TensorType:
    """
    Finds $x$ such that $Ax = b$, where $A$ is the hessian of model,
    and $b$ a vector.
    Depending on the inversion method, the hessian is either calculated directly
    and then inverted, or implicitly and then inverted through matrix vector
    product. The method also allows to add a small regularization term (lam)
    to facilitate inversion of non fully trained models.

    :param inversion_method:
    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param x: An array containing the features of the input data points.
    :param b:
    :param lam: regularization of the hessian
    :param progress: If True, display progress bars.

    :return: An array that solves the inverse problem,
        i.e. it returns $x$ such that $Ax = b$
    """
    if inversion_method == InversionMethod.Direct:
        return solve_linear(
            model.hessian(x, y, progress) + lam * identity_tensor(model.num_params()),
            b.T,
        ).T
    elif inversion_method == InversionMethod.Cg:
        grad_xy, _ = model.grad(x, y)
        backprop_on = model.parameters()
        reg_hvp = lambda v: mvp(grad_xy, v, backprop_on) + lam * v
        return solve_batch_cg(reg_hvp, b, progress)
    else:
        raise ValueError(f"Unknown inversion method: {inversion_method}")
