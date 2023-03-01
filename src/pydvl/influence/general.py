"""
Contains parallelized influence calculation functions for general models.
"""
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..utils import maybe_progress
from .frameworks import TensorType, TwiceDifferentiable
from .inversion_methods import InversionMethod, invert_matrix

__all__ = ["compute_influences", "InfluenceType", "calculate_influence_factors"]


class InfluenceType(str, Enum):
    """
    Different influence types.
    """

    Up = "up"
    Perturbation = "perturbation"


def calculate_influence_factors(
    model: TwiceDifferentiable,
    x: TensorType,
    y: TensorType,
    x_test: TensorType,
    y_test: TensorType,
    inversion_method: InversionMethod,
    lam: float = 0,
    progress: bool = False,
) -> NDArray:
    """
    It calculates influence factors of a model from its training and test
    points. Given a test point $z_test = (x_{test}, y_{test})$, a loss
    $L(z_{test}, \theta)$ ($\theta$ being the parameters of the model) and the
    Hessian of the model $H_{\theta}$, influence factors are defined as
    $s_{test} = H_{\theta}^{-1} \nabla_{\theta} L(z_{test}, \theta)$. They are
    used for efficient influence calculation. This method first
    (implicitly) calculates the Hessian and then (explicitly) finds the
    influence factors for the model using the given inversion method. The
    parameter ``lam`` is used to regularize the inversion of the Hessian. For
    more info, refer to :footcite:t:`koh_understanding_2017`, paragraph 3.

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param x_train: An array of shape [MxK] containing the features of the input data points.
    :param y_train: An array of shape [MxL] containing the targets of the input data points.
    :param x_test: An array of shape [NxK] containing the features of the test set of data points.
    :param y_test: An array of shape [NxL] containing the targets of the test set of data points.
    :param inversion_func: function to use to invert the product of hvp (hessian
        vector product) and the gradient of the loss (s_test in the paper).
    :param lam: regularization of the hessian
    :param progress: If True, display progress bars.
    :returns: An array of size (N, D) containing the influence factors for each
        dimension (D) and test sample (N).
    """
    grad_xy, _ = model.grad(x, y)
    hvp = lambda v: model.mvp(grad_xy, v) + lam * v
    n_params = model.num_params()
    test_grads = model.split_grad(x_test, y_test, progress)
    return invert_matrix(
        inversion_method,
        hvp,
        (n_params, n_params),
        test_grads,
        progress,
    )


def _calculate_influences_up(
    model: TwiceDifferentiable,
    x: TensorType,
    y: TensorType,
    influence_factors: NDArray,
    progress: bool = False,
) -> NDArray:
    """
    Given the model, the training points and the influence factors, calculates the
    influences using the upweighting method. More precisely, first it calculates
    the gradients of the model wrt. each training sample ($\nabla_{\theta} L$,
    with $L$ the loss of a single point and $\theta$ the parameters of the
    model) and then multiplies each with the influence factors. For more
    details, refer to section 2.1 of :footcite:t:`koh_understanding_2017`.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: An array of shape [MxK] containing the features of the
        input data points.
    :param y_train: An array of shape [MxL] containing the targets of the
        input data points.
    :param influence_factors: array containing influence factors
    :param progress: If True, display progress bars.
    :returns: An array of size [NxM], where N is number of test points and M
        number of train points.
    """
    train_grads = model.split_grad(x, y, progress)
    return np.einsum("ta,va->tv", influence_factors, train_grads)  # type: ignore


def _calculate_influences_pert(
    model: TwiceDifferentiable,
    x: TensorType,
    y: TensorType,
    influence_factors: NDArray,
    progress: bool = False,
) -> NDArray:
    """
    Calculates the influence values from the influence factors and the training
    points using the perturbation method. More precisely, for each training sample it
    calculates $\nabla_{\theta} L$ (with L the loss of the model over the single
    point and $\theta$ the parameters of the model) and then uses the method
    TwiceDifferentiable.mvp to efficiently calculate the product of the
    influence factors and $\nabla_x \nabla_{\theta} L$. For more details, refer
    to section 2.2 of :footcite:t:`koh_understanding_2017`.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: An array of shape [MxK] containing the features of the
        input data points.
    :param y_train: An array of shape [MxL] containing the targets of the
        input data points.
    :param influence_factors: array containing influence factors
    :param progress: If True, display progress bars.
    :returns: An array of size [NxMxP], where N is number of test points, M
        number of train points, and P the number of features.
    """
    all_pert_influences = []
    for i in maybe_progress(
        len(x),
        progress,
        desc="Influence Perturbation",
    ):
        grad_xy, tensor_x = model.grad(x[i : i + 1], y[i])
        perturbation_influences = model.mvp(
            grad_xy,
            influence_factors,
            backprop_on=tensor_x,
        )
        all_pert_influences.append(perturbation_influences.reshape((-1, *x[i].shape)))

    return np.stack(all_pert_influences, axis=1)


influence_type_registry = {
    InfluenceType.Up: _calculate_influences_up,
    InfluenceType.Perturbation: _calculate_influences_pert,
}


def compute_influences(
    differentiable_model: TwiceDifferentiable,
    x: TensorType,
    y: TensorType,
    x_test: TensorType,
    y_test: TensorType,
    progress: bool = False,
    inversion_method: InversionMethod = InversionMethod.Direct,
    influence_type: InfluenceType = InfluenceType.Up,
    hessian_regularization: float = 0,
) -> NDArray:
    """
    Calculates the influence of the training points j on the test points i.
    First it calculates the influence factors for all test points with respect
    to the training points, and then uses them to get the influences over the
    complete training set. Points with low influence values are (on average)
    less important for model training than points with high influences.

    :param differentiable_model: A model wrapped with its loss in TwiceDifferentiable.
    :param x: model input for training
    :param y: input labels
    :param x_test: model input for testing
    :param y_test: test labels
    :param progress: whether to display progress bars.
    :param inversion_method: Set the inversion method to a specific one, can be
        'direct' for direct inversion (and explicit construction of the Hessian)
        or 'cg' for conjugate gradient.
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

    influence_factors = calculate_influence_factors(
        differentiable_model,
        x,
        y,
        x_test,
        y_test,
        inversion_method,
        lam=hessian_regularization,
        progress=progress,
    )
    compute_influence_type = influence_type_registry[influence_type]

    return compute_influence_type(
        differentiable_model,
        x,
        y,
        influence_factors,
        progress,
    )
