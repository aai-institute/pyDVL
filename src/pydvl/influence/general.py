"""
This module contains parallelized influence calculation functions for general
models, as introduced in :footcite:t:`koh_understanding_2017`.
"""
import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, Generator, Optional, Type

from ..utils import maybe_progress
from .inversion import InverseHvpResult, InversionMethod, solve_hvp
from .twice_differentiable import (
    DataLoaderType,
    TensorType,
    TensorUtilities,
    TwiceDifferentiable,
)

__all__ = ["compute_influences", "InfluenceType", "compute_influence_factors"]

logger = logging.getLogger(__name__)


class InfluenceType(str, Enum):
    """
    Different influence types.
    """

    Up = "up"
    Perturbation = "perturbation"


def compute_influence_factors(
    model: TwiceDifferentiable,
    training_data: DataLoaderType,
    test_data: DataLoaderType,
    inversion_method: InversionMethod,
    *,
    hessian_perturbation: float = 0.0,
    progress: bool = False,
    **kwargs: Any,
) -> InverseHvpResult:
    r"""
    Calculates influence factors of a model for training and test
    data. Given a test point $z_test = (x_{test}, y_{test})$, a loss
    $L(z_{test}, \theta)$ ($\theta$ being the parameters of the model) and the
    Hessian of the model $H_{\theta}$, influence factors are defined as
    $$s_{test} = H_{\theta}^{-1} \grad_{\theta} L(z_{test}, \theta).$$. They are
    used for efficient influence calculation. This method first
    (implicitly) calculates the Hessian and then (explicitly) finds the
    influence factors for the model using the given inversion method. The
    parameter ``hessian_perturbation`` is used to regularize the inversion of
    the Hessian. For more info, refer to :footcite:t:`koh_understanding_2017`,
    paragraph 3.

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param test_data: A DataLoader containing the test data.
    :param inversion_method: name of method for computing inverse hessian vector
        products.
    :param hessian_perturbation: regularization of the hessian
    :param progress: If True, display progress bars.
    :returns: An array of size (N, D) containing the influence factors for each
        dimension (D) and test sample (N).
    """
    tensor_util: Type[TensorUtilities] = TensorUtilities.from_twice_differentiable(
        model
    )

    stack = tensor_util.stack
    unsqueeze = tensor_util.unsqueeze
    cat_gen = tensor_util.cat_gen
    cat = tensor_util.cat

    def test_grads() -> Generator[TensorType, None, None]:
        for x_test, y_test in maybe_progress(
            test_data, progress, desc="Batch Test Gradients"
        ):
            yield stack(
                [
                    model.grad(inpt, target)
                    for inpt, target in zip(unsqueeze(x_test, 1), y_test)
                ]
            )  # type:ignore

    try:
        # if provided input_data implements __len__, pre-allocate the result tensor to reduce memory consumption
        resulting_shape = (len(test_data), model.num_params)  # type:ignore
        rhs = cat_gen(
            test_grads(), resulting_shape, model  # type:ignore
        )  # type:ignore
    except Exception as e:
        logger.warning(
            f"Failed to pre-allocate result tensor: {e}\n"
            f"Evaluate all resulting tensor and concatenate"
        )
        rhs = cat(list(test_grads()))

    return solve_hvp(
        inversion_method,
        model,
        training_data,
        rhs,
        hessian_perturbation=hessian_perturbation,
        **kwargs,
    )


def compute_influences_up(
    model: TwiceDifferentiable,
    input_data: DataLoaderType,
    influence_factors: TensorType,
    *,
    progress: bool = False,
) -> TensorType:
    r"""
    Given the model, the training points and the influence factors, calculates the
    influences using the upweighting method. More precisely, first it calculates
    the gradients of the model wrt. each training sample ($\grad_{\theta} L$,
    with $L$ the loss of a single point and $\theta$ the parameters of the
    model) and then multiplies each with the influence factors. For more
    details, refer to section 2.1 of :footcite:t:`koh_understanding_2017`.

    :param model: A model which has to implement the TwiceDifferentiable
        interface.
    :param input_data: Data loader containing the samples to calculate the
        influence of.
    :param influence_factors: array containing influence factors
    :param progress: If True, display progress bars.
    :returns: An array of size [NxM], where N is number of influence factors, M
        number of input points.
    """

    tensor_util: Type[TensorUtilities] = TensorUtilities.from_twice_differentiable(
        model
    )

    stack = tensor_util.stack
    unsqueeze = tensor_util.unsqueeze
    cat_gen = tensor_util.cat_gen
    cat = tensor_util.cat
    einsum = tensor_util.einsum

    def train_grads() -> Generator[TensorType, None, None]:
        for x, y in maybe_progress(
            input_data, progress, desc="Batch Split Input Gradients"
        ):
            yield stack(
                [model.grad(inpt, target) for inpt, target in zip(unsqueeze(x, 1), y)]
            )  # type:ignore

    try:
        # if provided input_data implements __len__, pre-allocate the result tensor to reduce memory consumption
        resulting_shape = (len(input_data), model.num_params)  # type:ignore
        train_grad_tensor = cat_gen(
            train_grads(), resulting_shape, model  # type:ignore
        )  # type:ignore
    except Exception as e:
        logger.warning(
            f"Failed to pre-allocate result tensor: {e}\n"
            f"Evaluate all resulting tensor and concatenate"
        )
        train_grad_tensor = cat([x for x in train_grads()])  # type:ignore

    return einsum("ta,va->tv", influence_factors, train_grad_tensor)  # type:ignore


def compute_influences_pert(
    model: TwiceDifferentiable,
    input_data: DataLoaderType,
    influence_factors: TensorType,
    *,
    progress: bool = False,
) -> TensorType:
    r"""
    Calculates the influence values from the influence factors and the training
    points using the perturbation method. More precisely, for each training sample it
    calculates $\grad_{\theta} L$ (with L the loss of the model over the single
    point and $\theta$ the parameters of the model) and then uses the method
    TwiceDifferentiable.mvp to efficiently calculate the product of the
    influence factors and $\grad_x \grad_{\theta} L$. For more details, refer
    to section 2.2 of :footcite:t:`koh_understanding_2017`.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param input_data: Data loader containing the samples to calculate the
        influence of.
    :param influence_factors: array containing influence factors
    :param progress: If True, display progress bars.
    :returns: An array of size [NxMxP], where N is the number of influence factors, M
        the number of input data, and P the number of features.
    """

    tensor_util: Type[TensorUtilities] = TensorUtilities.from_twice_differentiable(
        model
    )
    stack = tensor_util.stack
    tu_slice = tensor_util.slice
    reshape = tensor_util.reshape
    get_element = tensor_util.get_element
    shape = tensor_util.shape

    all_pert_influences = []
    for x, y in maybe_progress(
        input_data,
        progress,
        desc="Batch Influence Perturbation",
    ):
        for i in range(len(x)):
            tensor_x = tu_slice(x, i, i + 1)
            grad_xy = model.grad(tensor_x, get_element(y, i), create_graph=True)
            perturbation_influences = model.mvp(
                grad_xy,
                influence_factors,
                backprop_on=tensor_x,
            )
            all_pert_influences.append(
                reshape(perturbation_influences, (-1, *shape(get_element(x, i))))
            )

    return stack(all_pert_influences, axis=1)  # type:ignore


influence_type_registry: Dict[InfluenceType, Callable[..., TensorType]] = {
    InfluenceType.Up: compute_influences_up,
    InfluenceType.Perturbation: compute_influences_pert,
}


def compute_influences(
    differentiable_model: TwiceDifferentiable,
    training_data: DataLoaderType,
    *,
    test_data: Optional[DataLoaderType] = None,
    input_data: Optional[DataLoaderType] = None,
    inversion_method: InversionMethod = InversionMethod.Direct,
    influence_type: InfluenceType = InfluenceType.Up,
    hessian_regularization: float = 0.0,
    progress: bool = False,
    **kwargs: Any,
) -> TensorType:  # type: ignore # ToDO fix typing
    r"""
    Calculates the influence of the input_data point j on the test points i.
    First it calculates the influence factors of all test points with respect
    to the model and the training points, and then uses them to get the
    influences over the complete input_data set.

    :param differentiable_model: A model wrapped with its loss in TwiceDifferentiable.
    :param training_data: data loader with the training data, used to calculate
        the hessian of the model loss.
    :param test_data: data loader with the test samples. If None, the samples in
        training_data are used.
    :param input_data: data loader with the samples to calculate the influences
        of. If None, the samples in training_data are used.
    :param progress: whether to display progress bars.
    :param influence_type: Which algorithm to use to calculate influences.
        Currently supported options: 'up' or 'perturbation'. For details refer
        to :footcite:t:`koh_understanding_2017`
    :param hessian_regularization: lambda to use in Hessian regularization, i.e.
        H_reg = H + lambda * 1, with 1 the identity matrix and H the (simple and
        regularized) Hessian. Typically used with more complex models to make
        sure the Hessian is positive definite.
    :returns: An array specifying the influences. Shape is [NxM] if
        influence_type is'up', where N is number of test points and M number of
        train points. If instead influence_type is 'perturbation', output shape
        is [NxMxP], with P the number of input features.
    """
    if input_data is None:
        input_data = deepcopy(training_data)
    if test_data is None:
        test_data = deepcopy(training_data)

    influence_factors, _ = compute_influence_factors(
        differentiable_model,
        training_data,
        test_data,
        inversion_method,
        hessian_perturbation=hessian_regularization,
        progress=progress,
        **kwargs,
    )

    return influence_type_registry[influence_type](
        differentiable_model,
        input_data,
        influence_factors,
        progress=progress,
    )
