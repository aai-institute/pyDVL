"""
Contains methods to invert the hessian vector product. 
"""
import logging
from enum import Enum
from typing import Any, Dict, Tuple

from .frameworks import (
    DataLoaderType,
    ModelType,
    TensorType,
    TwiceDifferentiable,
    solve_batch_cg,
    solve_linear,
    solve_lissa,
)

__all__ = ["solve_hvp"]

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"
    Lissa = "lissa"


def solve_hvp(
    inversion_method: InversionMethod,
    model: TwiceDifferentiable,
    training_data: DataLoaderType,
    b: TensorType,
    *,
    hessian_perturbation: float = 0.0,
    progress: bool = False,
    **kwargs: Any,
) -> Tuple[TensorType, Dict]:
    """
    Finds $x$ such that $Ax = b$, where $A$ is the hessian of model,
    and $b$ a vector.
    Depending on the inversion method, the hessian is either calculated directly
    and then inverted, or implicitly and then inverted through matrix vector
    product. The method also allows to add a small regularization term (hessian_perturbation)
    to facilitate inversion of non fully trained models.

    :param inversion_method:
    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param x: An array containing the features of the input data points.
    :param y: labels for x
    :param b: Array as the right hand side of the equation $Ax = b$
    :param kwargs: kwargs to pass to the inversion method
    :param hessian_perturbation: regularization of the hessian
    :param progress: If True, display progress bars.

    :return: An array that solves the inverse problem,
        i.e. it returns $x$ such that $Ax = b$, and a dictionary containing
        information about the inversion process.
    """
    if inversion_method == InversionMethod.Direct:
        return solve_linear(
            model,
            training_data,
            b,
            **kwargs,
            hessian_perturbation=hessian_perturbation,
            progress=progress,
        )
    elif inversion_method == InversionMethod.Cg:
        return solve_batch_cg(
            model,
            training_data,
            b,
            **kwargs,
            hessian_perturbation=hessian_perturbation,
            progress=progress,
        )
    elif inversion_method == InversionMethod.Lissa:
        return solve_lissa(
            model,
            training_data,
            b,
            **kwargs,
            hessian_perturbation=hessian_perturbation,
            progress=progress,
        )
    else:
        raise ValueError(f"Unknown inversion method: {inversion_method}")
