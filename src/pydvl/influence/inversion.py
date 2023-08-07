"""Contains methods to invert the hessian vector product.
"""
import functools
import inspect
import logging
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Tuple, Type, TypeVar

__all__ = ["solve_hvp", "InversionMethod", "InversionRegistry"]

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
    return InversionRegistry.call_registered(
        inversion_method,
        model,
        training_data,
        b,
        hessian_perturbation=hessian_perturbation,
        **kwargs,
    )


class InversionRegistry:
    """
    A registry to hold inversion methods for different models.
    """

    registry: Dict[Tuple[Type[TwiceDifferentiable], InversionMethod], Callable] = {}

    @classmethod
    def register(
        cls,
        model_type: Type[TwiceDifferentiable],
        inversion_method: InversionMethod,
        overwrite: bool = False,
    ):
        """
        Register a function for a specific model type and inversion method.

        The function to be registered must conform to the following signature:
        `(model: TwiceDifferentiable, training_data: DataLoaderType, b: TensorType, hessian_perturbation: float = 0.0, ...)`.

        :param model_type: The type of the model the function should be registered for.
        :param inversion_method: The inversion method the function should be registered for.
        :param overwrite: If True, allows overwriting of an existing registered function for the same model type and inversion method.
                      If False, logs a warning when attempting to register a function for an already registered model type and inversion method.


        :raises TypeError: If the provided model_type or inversion_method are of the wrong type.
        :raises ValueError: If the function to be registered does not match the required signature.
        :return: A decorator for registering a function.
        """

        if not isinstance(model_type, type):
            raise TypeError(f"'model_type' must be a Type[TwiceDifferentiable]")

        if not isinstance(inversion_method, InversionMethod):
            raise TypeError(f"'inversion_method' must be an InversionMethod")

        key = (model_type, inversion_method)

        def decorator(func):
            if not overwrite and key in cls.registry:
                warnings.warn(
                    f"There is already a function registered for model type {model_type} and inversion method {inversion_method}. "
                    f"To overwrite the existing function {cls.registry.get(key)} with {func}, set overwrite to True."
                )
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            expected_args = [
                ("model", model_type),
                ("training_data", DataLoaderType.__bound__),
                ("b", model_type.tensor_type()),
                ("hessian_perturbation", float),
            ]

            for (name, typ), param in zip(expected_args, params):
                if not (
                    isinstance(param.annotation, typ)
                    or issubclass(param.annotation, typ)
                ):
                    raise ValueError(
                        f'Parameter "{name}" must be of type "{typ.__name__}"'
                    )

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            cls.registry[key] = wrapper
            return wrapper

        return decorator

    @classmethod
    def get_registered(
        cls, model_type: Type[TwiceDifferentiable], inversion_method: InversionMethod
    ) -> Callable[
        [TwiceDifferentiable, DataLoaderType, TensorType, float], InverseHvpResult
    ]:
        key = (model_type, inversion_method)
        method = cls.registry.get(key, None)
        if method is None:
            raise ValueError(f"No function registered for {key}")
        return method

    @classmethod
    def call_registered(
        cls,
        inversion_method: InversionMethod,
        model: TwiceDifferentiable,
        training_data: DataLoaderType,
        b: TensorType,
        hessian_perturbation,
        **kwargs,
    ) -> InverseHvpResult:
        """
         Call a registered function with the provided parameters.

        :param inversion_method: The inversion method to use.
        :param model: A model wrapped in the TwiceDifferentiable interface.
        :param training_data: The training data to use.
        :param b: Array as the right hand side of the equation $Ax = b$.
        :param hessian_perturbation: Regularization of the hessian.
        :param kwargs: Additional keyword arguments to pass to the inversion method.

        :return: An object that contains an array that solves the inverse problem,
            i.e. it returns $x$ such that $Ax = b$, and a dictionary containing
            information about the inversion process.
        """

        return cls.get_registered(type(model), inversion_method)(
            model, training_data, b, hessian_perturbation, **kwargs
        )
