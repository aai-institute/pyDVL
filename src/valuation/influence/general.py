import functools
from typing import Callable, Dict, Optional

import numpy as np

from valuation.influence.types import InfluenceTypes
from valuation.solve.cg import conjugate_gradient
from valuation.utils import MapReduceJob, available_cpus, logger, map_reduce
from valuation.utils.types import (
    MatrixVectorProductInversionAlgorithm,
    TwiceDifferentiable,
)


def influences(
    model: TwiceDifferentiable,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    progress: bool = False,
    n_jobs: int = -1,
    influence_type: InfluenceTypes = InfluenceTypes.Up,
    inversion_method: str = None,
    max_data_points: int = None,
) -> np.ndarray:
    """
    Calculates the influence of the training points j on the test points i, with matrix I_(ij). It does so by
    calculating the influence factors for all test points, with respect to the training points. Subsequently,
    all influence get calculated over the train set.

    :param model: A model which has to implement the TwiceDifferentiable interface.
    :param x_train: A np.ndarray of shape [MxK] containing the features of the train set of data points.
    :param y_train: A np.ndarray of shape [MxL] containing the targets of the train set of data points.
    :param x_test: A np.ndarray of shape [NxK] containing the features of the test set of data points.
    :param y_test: A np.ndarray of shape [NxL] containing the targets of the test set of data points.

    :param progress: whether to display progress bars.
    :param n_jobs: The number of jobs to use for processing
    :param influence_type: Either InfluenceTypes.Up or InfluenceTypes.Perturbation.
    :param inversion_method: Set the inversion method to a specific one, can be either None for direct inversion
     (and explicit construction of the Hessian) or 'cg' for conjugate gradient.
    :returns: A np.ndarray of shape [NxM] specifying the influences.
    """

    n_params = model.num_params()
    dict_fact_algos: Dict[Optional[str], MatrixVectorProductInversionAlgorithm] = {
        None: lambda hvp, x: np.linalg.solve(hvp(np.eye(n_params)), x.T).T,
        "cg": lambda hvp, x: conjugate_gradient(hvp, x)[0],
    }

    cpu_count = available_cpus()
    if n_jobs == -1:
        n_jobs = cpu_count
    elif n_jobs <= 0:
        raise AttributeError(
            "The number of jobs has to b bigger than zero or -1 for all available cores."
        )

    if x_test is None or y_test is None:
        logger.info("No test data supplied, train data is reused.")
        x_test = x_train
        y_test = y_train

    # ------------------------------------------------------------------------------------------------------------------
    num_train_points = len(x_train)
    idx = None

    if max_data_points is not None:
        if max_data_points <= 0:
            raise AttributeError("Number of points have to bigger than 0.")

        real_num_data_points = min(num_train_points, max_data_points)
        idx = np.random.permutation(num_train_points)[:real_num_data_points]

    hvp_x_train = x_train if idx is None else x_train[idx]
    hvp_y_train = y_train if idx is None else y_train[idx]
    hvp = lambda v, **kwargs: model.mvp(
        hvp_x_train, hvp_y_train, v, progress=progress, **kwargs
    )

    def _calculate_influence_factors(indices: np.ndarray, job_id: int) -> np.ndarray:
        """
        Calculates the influence factors, e.g. the inverse Hessian vector products if the test cores with respect
        to the Hessian induced by all training points.

        :param indices: A np.ndarray containing all indices of the test data which shall be evaluated in this run.
        :param job_id: A id which describes the current job id.
        :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension and test sample.
        """
        c_x_test, c_y_test = x_test[indices], y_test[indices]
        test_grads = model.grad(c_x_test, c_y_test, progress=progress)
        return -1 * dict_fact_algos[inversion_method](hvp, test_grads)

    influence_factors_job = MapReduceJob.from_fun(
        _calculate_influence_factors, np.concatenate
    )
    if n_jobs == 1:
        influence_factors = _calculate_influence_factors(np.arange(len(x_test)), 0)
    else:
        influence_factors = map_reduce(
            influence_factors_job, np.arange(len(x_test)), num_jobs=n_jobs
        )[0]

    # ------------------------------------------------------------------------------------------------------------------

    def _calculate_influences_up(indices: np.ndarray, job_id: int) -> np.ndarray:
        """
        Calculates the influence from the influence factors and the scores of the training points.

        :param indices: A np.ndarray containing all indices of the training data which shall be evaluated in this run.
        :param job_id: A id which describes the current job id.
        :returns: A np.ndarray of size (N, K) containing the influence for each test sample and train sample.
        """
        c_x_train, c_y_train = x_train[indices], y_train[indices]
        train_grads = model.grad(c_x_train, c_y_train)
        return np.einsum("ta,va->tv", influence_factors, train_grads)

    def _calculate_influences_pert(indices: np.ndarray, job_id: int) -> np.ndarray:
        """
        Calculates the influence from the influence factors and the scores of the training points.

        :param indices: A np.ndarray containing all indices of the training data which shall be evaluated in this run.
        :param job_id: A id which describes the current job id.
        :returns: A np.ndarray of size (N, K) containing the influence for each test sample and train sample.
        """
        all_pert_influences = []
        for i in indices:
            perturbation_influences = model.mvp(
                x_train[i],
                y_train[i],
                influence_factors,
                progress=progress,
                second_x=True,
            )
            all_pert_influences.append(perturbation_influences)

        return np.stack(all_pert_influences, axis=1)

    influence_type_function_dict: Dict[
        InfluenceTypes, Callable[[np.ndarray, int], np.ndarray]
    ] = {
        InfluenceTypes.Up: _calculate_influences_up,
        InfluenceTypes.Perturbation: _calculate_influences_pert,
    }
    influence_function = influence_type_function_dict[influence_type]
    if n_jobs == 1:
        return influence_function(np.arange(len(x_train)), 0)

    influences_job = MapReduceJob.from_fun(
        influence_function, functools.partial(np.concatenate, axis=1)
    )
    return map_reduce(influences_job, np.arange(len(x_train)), num_jobs=n_jobs)[0]
