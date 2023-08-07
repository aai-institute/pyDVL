"""Contains methods to invert the hessian vector product.
"""
import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Tuple, Type, TypeVar

__all__ = ["solve_hvp", "InversionMethod", "inversion_registry"]

from .frameworks.twice_differentiable import (
    InverseHvpResult,
    TensorType,
    TwiceDifferentiable,
)

logger = logging.getLogger(__name__)

DataLoaderType = TypeVar("DataLoaderType", bound=Iterable)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"
    Lissa = "lissa"
    Arnoldi = "arnoldi"


def solve_hvp(
    inversion_method: InversionMethod,
    model: TwiceDifferentiable,
    training_data: DataLoaderType,
    b: TensorType,
    *,
    hessian_perturbation: float = 0.0,
    **kwargs: Any,
) -> InverseHvpResult:
    """
    Finds $x$ such that $Ax = b$, where $A$ is the hessian of model,
    and $b$ a vector.
    Depending on the inversion method, the hessian is either calculated directly
    and then inverted, or implicitly and then inverted through matrix vector
    product. The method also allows to add a small regularization term (hessian_perturbation)
    to facilitate inversion of non fully trained models.

    :param inversion_method:
    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data:
    :param b: Array as the right hand side of the equation $Ax = b$
    :param hessian_perturbation: regularization of the hessian
    :param kwargs: kwargs to pass to the inversion method

    :return: An object that contains an array that solves the inverse problem,
        i.e. it returns $x$ such that $Ax = b$, and a dictionary containing
        information about the inversion process.
    """
    method_implementation = inversion_registry.get(
        (type(model), inversion_method), None
    )

    if method_implementation is None:
        raise ValueError(
            f"No implementation found for model type {type(model)} and method {inversion_method}"
        )

    return method_implementation(
        model,
        training_data,
        b,
        hessian_perturbation,
        **kwargs,
    )


def check_function_signature(
    function: Callable[..., InverseHvpResult], model_type: Type[TwiceDifferentiable]
):
    sig = inspect.signature(function)
    params = list(sig.parameters.values())

    expected_args = [
        ("model", model_type),
        ("training_data", DataLoaderType.__bound__),  # type: ignore # ToDO fix typing
        ("b", model_type.tensor_type()),
        ("hessian_perturbation", float),
    ]

    for (name, typ), param in zip(expected_args, params):
        if not (isinstance(param.annotation, typ) or issubclass(param.annotation, typ)):
            raise ValueError(f'Parameter "{name}" must be of type "{typ.__name__}"')


inversion_registry: Dict[
    Tuple[Type[TwiceDifferentiable], InversionMethod],
    Callable[..., InverseHvpResult],
] = {}

# For implementing new frameworks and or add new methods, extend this code block accordingly
try:

    import torch

    from .frameworks import (
        TorchTwiceDifferentiable,
        solve_arnoldi,
        solve_batch_cg,
        solve_linear,
        solve_lissa,
    )

    method_name_to_impl: Dict[
        InversionMethod,
        Callable[
            [TorchTwiceDifferentiable, DataLoaderType, torch.Tensor, float],
            InverseHvpResult,
        ],
    ] = {
        InversionMethod.Direct: solve_linear,
        InversionMethod.Lissa: solve_lissa,
        InversionMethod.Cg: solve_batch_cg,
        InversionMethod.Arnoldi: solve_arnoldi,
    }

    for method, func in method_name_to_impl.items():
        check_function_signature(func, TorchTwiceDifferentiable)
        inversion_registry[(TorchTwiceDifferentiable, method)] = func

except ImportError:
    pass
