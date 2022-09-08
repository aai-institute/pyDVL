"""
Contains parallelized influence calculation functions for general models.
"""

__all__ = ["influences"]

from typing import Dict, Optional

import numpy as np

from valuation.influence.cg import (
    batched_preconditioned_conjugate_gradient,
    hvp_to_inv_diag_conditioner,
)
from valuation.utils import Dataset
from valuation.utils.types import (
    MatrixVectorProductInversionAlgorithm,
    TwiceDifferentiable,
)


def calculate_influence_factors(
    model: TwiceDifferentiable,
    data: Dataset,
    inversion_func: MatrixVectorProductInversionAlgorithm,
    train_indices: Optional[np.array] = None,
    test_indices: Optional[np.array] = None,
    progress: bool = False,
) -> np.ndarray:
    """
    Calculates the influence factors. For more info, see https://arxiv.org/pdf/1703.04730.pdf, paragraph 3.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param data: a dataset
    :param train_indices: which train indices to calculate the influence factors of
    :param test_indices: which test indices to use to calculate the influence factors
    :param inversion_func: function to use to invert the the product of hvp (hessian vector product) and the gradient
        of the loss (s_test in the paper).
    :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension (D) and test sample (N).
    """
    x_train, y_train = data.get_train_data(train_indices)
    x_test, y_test = data.get_test_data(test_indices)

    hvp = lambda v, **kwargs: model.mvp(
        x_train, y_train, v, progress=progress, **kwargs
    )
    test_grads = model.grad(x_test, y_test, progress=progress)
    return -1 * inversion_func(hvp, test_grads)


def _calculate_influences_up(
    model: TwiceDifferentiable,
    data: Dataset,
    influence_factors: np.ndarray,
    train_indices: Optional[np.array] = None,
) -> np.ndarray:
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the upweighting method, as described in section 2.1 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param data: a dataset
    :param train_indices: which train indices to calculate the influence factors of
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxM], where N is number of test points and M number of train points.
    """
    x_train, y_train = data.get_train_data(train_indices)
    train_grads = model.grad(x_train, y_train)
    return np.einsum("ta,va->tv", influence_factors, train_grads)  # type: ignore


def _calculate_influences_pert(
    model: TwiceDifferentiable,
    data: Dataset,
    influence_factors: np.ndarray,
    train_indices: Optional[np.array] = None,
) -> np.ndarray:
    """
    Calculates the influence from the influence factors and the scores of the training points.
    Uses the perturbation method, as described in section 2.2 of https://arxiv.org/pdf/1703.04730.pdf

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param data: a dataset
    :param train_indices: which train indices to calculate the influence factors of
    :param influence_factors: np.ndarray containing influence factors
    :returns: A np.ndarray of size [NxM], where N is number of test points and M number of train points.
    """
    all_pert_influences = []
    x_train, y_train = data.get_train_data(train_indices)
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
    model: TwiceDifferentiable,
    data: Dataset,
    progress: bool = False,
    inversion_method: str = "direct",
    influence_type: str = "up",
    train_points_idxs: Optional[int] = None,
) -> np.ndarray:
    """
    Calculates the influence of the training points j on the test points i, with matrix I_(ij). It does so by
    calculating the influence factors for all test points, with respect to the training points. Subsequently,
    all influence get calculated over the complete train set.

    :param model: A model following the TwiceDifferentiable interface.
    :param data: a dataset

    :param progress: whether to display progress bars.
    :param inversion_method: Set the inversion method to a specific one, can be 'direct' for direct inversion
        (and explicit construction of the Hessian) or 'cg' for conjugate gradient.
    :param influence_type: Which algorithm to use to calculate influences.
        Currently supported options: 'up' or 'perturbation'
    :param train_points_idxs: It indicates which train data points to calculate the influence score of.
        If None, it calculates influences for all training data points.
    :returns: A np.ndarray specifying the influences. Shape is [NxM], where N is number of test points and
        M number of train points.
    """

    n_params = model.num_params()
    dict_fact_algos: Dict[Optional[str], MatrixVectorProductInversionAlgorithm] = {
        "direct": lambda hvp, x: np.linalg.solve(hvp(np.eye(n_params)), x.T).T,  # type: ignore
        "cg": lambda hvp, x: batched_preconditioned_conjugate_gradient(  # type: ignore
            hvp, x, M=hvp_to_inv_diag_conditioner(hvp, d=x.shape[1])
        )[0],
    }

    influence_factors = calculate_influence_factors(
        model,
        data,
        dict_fact_algos[inversion_method],
        train_indices=train_points_idxs,
        progress=progress,
    )
    influence_function = influence_type_function_dict[influence_type]

    return influence_function(model, data, influence_factors)
