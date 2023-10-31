"""Contains methods to invert the hessian vector product.
"""
import functools
import inspect
import logging
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Tuple, Type

__all__ = [
    "solve_hvp",
    "InversionMethod",
    "InversionRegistry",
]

from .twice_differentiable import (
    DataLoaderType,
    InverseHvpResult,
    TensorType,
    TwiceDifferentiable,
)

logger = logging.getLogger(__name__)


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"
    Lissa = "lissa"
    Arnoldi = "arnoldi"
    Ekfac = "ekfac"


def solve_hvp(
    inversion_method: InversionMethod,
    model: TwiceDifferentiable,
    training_data: DataLoaderType,
    b: TensorType,
    *,
    hessian_perturbation: float = 0.0,
    **kwargs: Any,
) -> InverseHvpResult:
    r"""
    Finds \( x \) such that \( Ax = b \), where \( A \) is the hessian of the model,
    and \( b \) a vector. Depending on the inversion method, the hessian is either
    calculated directly and then inverted, or implicitly and then inverted through
    matrix vector product. The method also allows to add a small regularization term
    (hessian_perturbation) to facilitate inversion of non fully trained models.

    Args:
        inversion_method:
        model: A model wrapped in the TwiceDifferentiable interface.
        training_data:
        b: Array as the right hand side of the equation \( Ax = b \)
        hessian_perturbation: regularization of the hessian.
        kwargs: kwargs to pass to the inversion method.

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult], with
            an array that solves the inverse problem, i.e., it returns \( x \) such that \( Ax = b \)
            and a dictionary containing information about the inversion process.
    """

    return InversionRegistry.call(
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
        `(model: TwiceDifferentiable, training_data: DataLoaderType, b: TensorType,
        hessian_perturbation: float = 0.0, ...)`.

        Args:
            model_type: The type of the model the function should be registered for.
            inversion_method: The inversion method the function should be
                registered for.
            overwrite: If ``True``, allows overwriting of an existing registered
                function for the same model type and inversion method. If ``False``,
                logs a warning when attempting to register a function for an already
                registered model type and inversion method.

        Raises:
            TypeError: If the provided model_type or inversion_method are of the wrong type.
            ValueError: If the function to be registered does not match the required signature.

        Returns:
            A decorator for registering a function.
        """

        if not isinstance(model_type, type):
            raise TypeError(
                f"'model_type' is of type {type(model_type)} but should be a Type[TwiceDifferentiable]"
            )

        if not isinstance(inversion_method, InversionMethod):
            raise TypeError(
                f"'inversion_method' must be an 'InversionMethod' "
                f"but has type {type(inversion_method)} instead."
            )

        key = (model_type, inversion_method)

        def decorator(func):
            if not overwrite and key in cls.registry:
                warnings.warn(
                    f"There is already a function registered for model type {model_type} "
                    f"and inversion method {inversion_method}. "
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
    def get(
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
    def call(
        cls,
        inversion_method: InversionMethod,
        model: TwiceDifferentiable,
        training_data: DataLoaderType,
        b: TensorType,
        hessian_perturbation,
        **kwargs,
    ) -> InverseHvpResult:
        r"""
        Call a registered function with the provided parameters.

        Args:
            inversion_method: The inversion method to use.
            model: A model wrapped in the TwiceDifferentiable interface.
            training_data: The training data to use.
            b: Array as the right hand side of the equation \(Ax = b\).
            hessian_perturbation: Regularization of the hessian.
            kwargs: Additional keyword arguments to pass to the inversion method.

        Returns:
            An instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult],
                that contains an array, which solves the inverse problem,
                i.e. it returns \(x\) such that \(Ax = b\), and a dictionary containing information
                about the inversion process.
        """

        return cls.get(type(model), inversion_method)(
            model, training_data, b, hessian_perturbation, **kwargs
        )
