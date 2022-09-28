"""
Contains parallelized influence calculation functions for general models.
"""
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Optional

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

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def calculate_influence_factors(
    model: TwiceDifferentiable,
    x_train: "NDArray",
    y_train: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
    inversion_func: MatrixVectorProductInversionAlgorithm,
    *,
    progress: bool = False,
) -> np.ndarray:
    """
    Calculates the influence factors. For more info, see https://arxiv.org/pdf/1703.04730.pdf, paragraph 3.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :param inversion_func: function to use to invert the product of hvp (hessian vector product) and the gradient
        of the loss (s_test in the paper).
    :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension (D) and test sample (N).
    :param progress: If True, display progress bars.
    """

    hvp = lambda v, **kwargs: model.mvp(
        x_train, y_train, v, progress=progress, **kwargs
    )
    test_grads = model.grad(x_test, y_test, progress=progress)
    return -1 * inversion_func(hvp, test_grads)


def _calculate_influences_up(
    model: TwiceDifferentiable,
    x_train: "NDArray",
    y_train: "NDArray",
    influence_factors: "NDArray",
) -> np.ndarray:
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the upweighting method, as described in section 2.1 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxM], where N is number of test points and M number of train points.
    """
    train_grads = model.grad(x_train, y_train)
    return np.einsum("ta,va->tv", influence_factors, train_grads)  # type: ignore


def _calculate_influences_pert(
    model: TwiceDifferentiable,
    x_train: "NDArray",
    y_train: "NDArray",
    influence_factors: "NDArray",
) -> "NDArray":
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the perturbation method, as described in section 2.2 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxM], where N is number of test points and M number of train points.
    """
    all_pert_influences = []
    for i in np.arange(len(x_train)):
        perturbation_influences = model.mvp(
            x_train[i],
            y_train[i],
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
    x_train: "NDArray",
    y_train: "NDArray",
    x_test: "NDArray",
    y_test: "NDArray",
    *,
    inversion_method: InversionMethod = InversionMethod.Direct,
    influence_type: InfluenceType = InfluenceType.Up,
    inversion_method_kwargs: Optional[Dict] = None,
    progress: bool = False,
) -> "NDArray":
    """
    Calculates the influence of the training points j on the test points i. First it calculates
    the influence factors for all test points with respect to the training points, and then uses them to
    get the influences over the complete training set. Points with low influence values are (on average)
    less important for model training than points with high influences.

    :param model: A supervised model from a supported framework. Currently, only pytorch nn.Module is supported.
    :param loss: Loss function.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.
    :param inversion_method: Set the inversion method to a specific one, can be 'direct' for direct inversion \
        (and explicit construction of the Hessian) or 'cg' for conjugate gradient.
    :param influence_type: Which algorithm to use to calculate influences. \
        Currently supported options: 'up' or 'perturbation'
    :param inversion_method_kwargs: Keyword arguments for the influence method.
    :returns: A np.ndarray specifying the influences. Shape is [NxM], where N is number of test points and \
        M number of train points.
    :param progress: If True, display progress bars.
    """
    if inversion_method_kwargs is None:
        inversion_method_kwargs = dict()

    differentiable_model = TorchTwiceDifferentiable(model, loss)
    n_params = differentiable_model.num_params()
    dict_fact_algos: Dict[Optional[str], MatrixVectorProductInversionAlgorithm] = {
        "direct": lambda hvp, x: np.linalg.solve(hvp(np.eye(n_params)), x.T).T,  # type: ignore
        "cg": lambda hvp, x: batched_preconditioned_conjugate_gradient(  # type: ignore
            hvp,
            x,
            M=hvp_to_inv_diag_conditioner(hvp, d=x.shape[1]),
            **inversion_method_kwargs,
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
