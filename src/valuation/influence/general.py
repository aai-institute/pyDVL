"""
Contains parallelized influence calculation functions for general models.
"""

from enum import Enum
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from valuation.influence.conjugate_gradient import (
    batched_preconditioned_conjugate_gradient,
    conjugate_gradient,
)
from valuation.influence.frameworks import TorchTwiceDifferentiable
from valuation.influence.types import (
    MatrixVectorProductInversionAlgorithm,
    TwiceDifferentiable,
)

__all__ = ["influences", "InfluenceType", "InversionMethod"]


class InfluenceType(str, Enum):
    """
    Different influence types.
    """

    Up = "up"
    Perturbation = "perturbation"


class InversionMethod(str, Enum):
    """
    Different inversion methods types.
    """

    Direct = "direct"
    Cg = "cg"
    BatchedCg = "batched_cg"


def calculate_influence_factors(
    model: TwiceDifferentiable,
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    inversion_func: MatrixVectorProductInversionAlgorithm,
    lam=0,
    progress: bool = False,
) -> np.ndarray:
    """
    Calculates the influence factors. For more info, see https://arxiv.org/pdf/1703.04730.pdf, paragraph 3.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x: model input for training
    :param y: input labels
    :param x_test: model input for testing
    :param y_test: test labels
    :param inversion_func: function to use to invert the hvp (hessian vector product) and the gradient
        of the loss (s_test in the paper).
    :param progress: True for plotting the progress bar, False otherwise.
    :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension (D) and test sample (N).
    """
    grad_xy, _ = model.grad(x, y)
    hvp = lambda v: model.mvp(grad_xy, v, progress=progress) + lam * v
    test_grads = model.split_grad(x_test, y_test, progress=progress)
    return inversion_func(hvp, test_grads)


def _calculate_influences_up(
    model: TwiceDifferentiable,
    x: np.ndarray,
    y: np.ndarray,
    influence_factors: np.ndarray,
) -> np.ndarray:
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the upweighting method, as described in section 2.1 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x: input of the model
    :param y: labels
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxM], where N is number of test points and M number of train points.
    """
    train_grads = model.split_grad(x, y)
    return np.einsum("ta,va->tv", influence_factors, train_grads)  # type: ignore


def _calculate_influences_pert(
    model: TwiceDifferentiable,
    x: np.ndarray,
    y: np.ndarray,
    influence_factors: np.ndarray,
) -> np.ndarray:
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the perturbation method, as described in section 2.2 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x: input of the model
    :param y: labels
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxMxP], where N is number of test points, M number of train points,
        and P the number of features.
    """
    all_pert_influences = []
    for i in np.arange(len(x)):
        grad_xy, tensor_x = model.grad(x[i], y[i])
        perturbation_influences = model.mvp(
            grad_xy,
            influence_factors,
            backprop_on=[tensor_x],
        )
        all_pert_influences.append(perturbation_influences)

    return np.stack(all_pert_influences, axis=1)


influence_type_function_dict = {
    "up": _calculate_influences_up,
    "perturbation": _calculate_influences_pert,
}


def influences(
    model: nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    progress: bool = False,
    inversion_method: InversionMethod = InversionMethod.Direct,
    influence_type: InfluenceType = InfluenceType.Up,
    inversion_method_kwargs: Optional[Dict] = None,
    lam=0,
) -> np.ndarray:
    """
    Calculates the influence of the training points j on the test points i. First it calculates
    the influence factors for all test points with respect to the training points, and then uses them to
    get the influences over the complete training set. Points with low influence values are (on average)
    less important for model training than points with high influences.

    :param model: A supervised model from a supported framework. Currently, only pytorch nn.Module is supported.
    :param loss: loss of the model, a callable that, given prediction of the model and real labels, returns a
        tensor with the loss value.
    :param x: model input for training
    :param y: input labels
    :param x_test: model input for testing
    :param y_test: test labels
    :param progress: whether to display progress bars.
    :param inversion_method: Set the inversion method to a specific one, can be 'direct' for direct inversion
        (and explicit construction of the Hessian) or 'cg' for conjugate gradient.
    :param influence_type: Which algorithm to use to calculate influences.
        Currently supported options: 'up' or 'perturbation'. For details refer to https://arxiv.org/pdf/1703.04730.pdf
    :param inversion_method_kwargs: kwargs for the inversion method selected.
        If using the direct method no kwargs are needed. If inversion_method='cg', the following kwargs can be passed:
        - rtol: relative tolerance to be achieved before terminating computation
        - max_iterations: maximum conjugate gradient iterations
        - max_step_size: step size of conjugate gradient
        - verify_assumptions: True to run tests on convexity of the model.
    :returns: A np.ndarray specifying the influences. Shape is [NxM] if influence_type is'up', where N is number of test points and
        M number of train points. If instead influence_type is 'perturbation', output shape is [NxMxP], with P the number of input
        features.
    """
    if inversion_method_kwargs is None:
        inversion_method_kwargs = {}
    differentiable_model = TorchTwiceDifferentiable(model, loss)
    n_params = differentiable_model.num_params()
    dict_fact_algos: Dict[Optional[str], MatrixVectorProductInversionAlgorithm] = {
        "direct": lambda hvp, x: np.linalg.solve(hvp(np.eye(n_params)), x.T).T,  # type: ignore
        "cg": lambda hvp, x: conjugate_gradient(hvp(np.eye(n_params)), x),
        "batched_cg": lambda hvp, x: batched_preconditioned_conjugate_gradient(  # type: ignore
            hvp, x, **inversion_method_kwargs
        )[
            0
        ],
    }

    influence_factors = calculate_influence_factors(
        differentiable_model,
        x,
        y,
        x_test,
        y_test,
        dict_fact_algos[inversion_method],
        lam=lam,
        progress=progress,
    )
    influence_function = influence_type_function_dict[influence_type]

    return influence_function(
        differentiable_model,
        x,
        y,
        influence_factors,
    )
