"""
Contains parallelized influence calculation functions for general models.
"""

__all__ = ["influences"]

from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from valuation.influence.cg import (
    batched_preconditioned_conjugate_gradient,
    hvp_to_inv_diag_conditioner,
)
from valuation.influence.frameworks import TorchTwiceDifferentiable
from valuation.influence.types import (
    MatrixVectorProductInversionAlgorithm,
    TwiceDifferentiable,
)
from valuation.utils import Dataset


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


def calculate_influence_factors(
    model: TwiceDifferentiable,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    inversion_func: MatrixVectorProductInversionAlgorithm,
    progress: bool = False,
) -> np.ndarray:
    """
    Calculates the influence factors. For more info, see https://arxiv.org/pdf/1703.04730.pdf, paragraph 3.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param data: a dataset
    :param train_indices: which train indices to calculate the influence factors of
    :param test_indices: which test indices to use to calculate the influence factors
    :param inversion_func: function to use to invert the product of hvp (hessian vector product) and the gradient
        of the loss (s_test in the paper).
    :param progress: True for plotting the progress bar, False otherwise.
    :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension (D) and test sample (N).
    """

    hvp = lambda v, **kwargs: model.mvp(
        x_train, y_train, v, progress=progress, **kwargs
    )
    test_grads = model.grad(x_test, y_test, progress=progress)
    return -1 * inversion_func(hvp, test_grads)


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
    train_grads = model.grad(x, y)
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
        perturbation_influences = model.mvp(
            x[i],
            y[i],
            influence_factors,
            second_x=True,
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
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    progress: bool = False,
    inversion_method: InversionMethod = InversionMethod.Direct,
    influence_type: InfluenceType = InfluenceType.Up,
    inversion_method_kwargs: Dict = {},
) -> np.ndarray:
    """
    Calculates the influence of the training points j on the test points i. First it calculates
    the influence factors for all test points with respect to the training points, and then uses them to
    get the influences over the complete training set. Points with low influence values are (on average)
    less important for model training than points with high influences.

    :param model: A supervised model from a supported framework. Currently, only pytorch nn.Module is supported.
    :param loss: loss of the model, a callable that, given prediction of the model and real labels, returns a
        tensor with the loss value.
    :param x_train: model input for training
    :param y_train: train labels
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
        - raise_exception: True for raising error if assumptions are not met. If false, warning will be logged.
    :returns: A np.ndarray specifying the influences. Shape is [NxM] if influence_type is'up', where N is number of test points and
        M number of train points. If instead influence_type is 'perturbation', output shape is [NxMxP], with P the number of input
        features.
    """
    differentiable_model = TorchTwiceDifferentiable(model, loss)
    n_params = differentiable_model.num_params()
    dict_fact_algos: Dict[Optional[str], MatrixVectorProductInversionAlgorithm] = {
        "direct": lambda hvp, x: np.linalg.solve(hvp(np.eye(n_params)), x.T).T,  # type: ignore
        "cg": lambda hvp, x: batched_preconditioned_conjugate_gradient(  # type: ignore
            hvp,
            x,
            M=hvp_to_inv_diag_conditioner(hvp, d=x.shape[1]),
            **inversion_method_kwargs
        )[0],
    }

    influence_factors = calculate_influence_factors(
        differentiable_model,
        x_train,
        y_train,
        x_test,
        y_test,
        dict_fact_algos[inversion_method],
        progress=progress,
    )
    influence_function = influence_type_function_dict[influence_type]

    # The -1 here is to have increasing influence for better quality points.
    # It could be simplified with the -1 in the influence factors definition,
    # but to keep definition consistent with the original paper we flip sign here.
    return -1 * influence_function(
        differentiable_model,
        x_train,
        y_train,
        influence_factors,
    )
