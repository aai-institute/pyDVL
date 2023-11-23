"""
This module contains influence calculation functions for general
models, as introduced in (Koh and Liang, 2017)[^1].

## References

[^1]: <a name="koh_liang_2017"></a>Koh, P.W., Liang, P., 2017.
    [Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html).
    In: Proceedings of the 34th International Conference on Machine Learning, pp. 1885–1894. PMLR.
"""
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, Optional, Type

from ..utils import maybe_progress
from .base_influence_model import InfluenceType, TensorType
from .inversion import InfluenceRegistry, InversionMethod
from .twice_differentiable import (
    DataLoaderType,
    InverseHvpResult,
    TensorUtilities,
    TwiceDifferentiable,
)

__all__ = ["compute_influences", "compute_influence_factors"]

logger = logging.getLogger(__name__)


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
    Calculates influence factors of a model for training and test data.

    Given a test point \(z_{test} = (x_{test}, y_{test})\), a loss \(L(z_{test}, \theta)\)
    (\(\theta\) being the parameters of the model) and the Hessian of the model \(H_{\theta}\),
    influence factors are defined as:

    \[
    s_{test} = H_{\theta}^{-1} \operatorname{grad}_{\theta} L(z_{test}, \theta).
    \]

    They are used for efficient influence calculation. This method first (implicitly) calculates
    the Hessian and then (explicitly) finds the influence factors for the model using the given
    inversion method. The parameter `hessian_perturbation` is used to regularize the inversion of
    the Hessian. For more info, refer to (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>, paragraph 3.

    Args:
        model: A model wrapped in the TwiceDifferentiable interface.
        training_data: DataLoader containing the training data.
        test_data: DataLoader containing the test data.
        inversion_method: Name of method for computing inverse hessian vector products.
        hessian_perturbation: Regularization of the hessian.
        progress: If True, display progress bars.

    Returns:
        array: An array of size (N, D) containing the influence factors for each dimension (D) and test sample (N).

    """

    tensor_util: Type[TensorUtilities] = TensorUtilities.from_twice_differentiable(
        model
    )
    cat = tensor_util.cat

    influence = InfluenceRegistry.get(  # type:ignore
        type(model), inversion_method
    )(
        model, training_data, hessian_perturbation, **kwargs
    )

    def factors_gen() -> Generator[TensorType, None, None]:
        for x_test, y_test in maybe_progress(
            test_data, progress, desc="Batch test factors"
        ):
            yield influence.factors(x_test, y_test)

    return cat(list(factors_gen()))


def compute_influences_up(
    model: TwiceDifferentiable,
    input_data: DataLoaderType,
    influence_factors: TensorType,
    *,
    progress: bool = False,
) -> TensorType:
    r"""
    Given the model, the training points, and the influence factors, this function calculates the
    influences using the up-weighting method.

    The procedure involves two main steps:
    1. Calculating the gradients of the model with respect to each training sample
       (\(\operatorname{grad}_{\theta} L\), where \(L\) is the loss of a single point and \(\theta\) are the
       parameters of the model).
    2. Multiplying each gradient with the influence factors.

    For a detailed description of the methodology, see section 2.1 of (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>.

    Args:
        model: A model that implements the TwiceDifferentiable interface.
        input_data: DataLoader containing the samples for which the influence will be calculated.
        influence_factors: Array containing pre-computed influence factors.
        progress: If set to True, progress bars will be displayed during computation.

    Returns:
        An array of shape [NxM], where N is the number of influence factors, and M is the number of input samples.
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
        # in case input_data is a torch DataLoader created from a Dataset,
        # we can pre-allocate the result tensor to reduce memory consumption
        resulting_shape = (len(input_data.dataset), model.num_params)  # type:ignore
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
    Calculates the influence values based on the influence factors and training samples using the perturbation method.

    The process involves two main steps:
    1. Calculating the gradient of the model with respect to each training sample
       (\(\operatorname{grad}_{\theta} L\), where \(L\) is the loss of the model for a single data point and \(\theta\)
       are the parameters of the model).
    2. Using the method [TwiceDifferentiable.mvp][pydvl.influence.twice_differentiable.TwiceDifferentiable.mvp]
       to efficiently compute the product of the
       influence factors and \(\operatorname{grad}_x \operatorname{grad}_{\theta} L\).

    For a detailed methodology, see section 2.2 of (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>.

    Args:
        model: A model that implements the TwiceDifferentiable interface.
        input_data: DataLoader containing the samples for which the influence will be calculated.
        influence_factors: Array containing pre-computed influence factors.
        progress: If set to True, progress bars will be displayed during computation.

    Returns:
        A 3D array with shape [NxMxP], where N is the number of influence factors,
            M is the number of input samples, and P is the number of features.
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
    Calculates the influence of each input data point on the specified test points.

    This method operates in two primary stages:
    1. Computes the influence factors for all test points concerning the model and its training data.
    2. Uses these factors to derive the influences over the complete set of input data.

    The influence calculation relies on the twice-differentiable nature of the provided model.

    Args:
        differentiable_model: A model bundled with its corresponding loss in the `TwiceDifferentiable` wrapper.
        training_data: DataLoader instance supplying the training data. This data is pivotal in computing the
                       Hessian matrix for the model's loss.
        test_data: DataLoader instance with the test samples. Defaults to `training_data` if None.
        input_data: DataLoader instance holding samples whose influences need to be computed. Defaults to
                    `training_data` if None.
        inversion_method: An enumeration value determining the approach for inverting matrices
            or computing inverse operations, see [.inversion.InversionMethod]
        progress: A boolean indicating whether progress bars should be displayed during computation.
        influence_type: Determines the methodology for computing influences.
            Valid choices include 'up' (for up-weighting) and 'perturbation'.
            For an in-depth understanding, see (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>.
        hessian_regularization: A lambda value used in Hessian regularization. The regularized Hessian, \( H_{reg} \),
            is computed as \( H + \lambda \times I \), where \( I \) is the identity matrix and \( H \)
            is the simple, unmodified Hessian. This regularization is typically utilized for more
            sophisticated models to ensure that the Hessian remains positive definite.

    Returns:
        The shape of this array varies based on the `influence_type`. If 'up', the shape is [NxM], where
            N denotes the number of test points and M denotes the number of training points. Conversely, if the
            influence_type is 'perturbation', the shape is [NxMxP], with P representing the number of input features.
    """

    if input_data is None:
        input_data = deepcopy(training_data)
    if test_data is None:
        test_data = deepcopy(training_data)

    factors = compute_influence_factors(
        differentiable_model,
        training_data,
        test_data,
        inversion_method,
        hessian_perturbation=hessian_regularization,
        **kwargs,
    )

    influence = InfluenceRegistry.get(  # type:ignore
        type(differentiable_model), inversion_method
    )(differentiable_model, training_data, hessian_regularization, **kwargs)

    influence_function = (
        influence.up_weighting
        if influence_type is InfluenceType.Up
        else influence.perturbation
    )

    def values_gen() -> Generator[TensorType, None, None]:
        for x, y in maybe_progress(
            input_data, progress, desc="Batch input influence values"
        ):
            yield influence_function(factors, x, y)

    tensor_util: Type[TensorUtilities] = TensorUtilities.from_twice_differentiable(
        differentiable_model
    )
    cat = tensor_util.cat
    values = cat(list(values_gen()), dim=1)

    return values
