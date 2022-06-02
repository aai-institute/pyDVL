import functools
import multiprocessing

import numpy as np
from opt_einsum import contract

from valuation.models.pytorch_model import TwiceDifferentiable
from valuation.utils import (
    Dataset,
    MapReduceJob,
    SupervisedModel,
    Utility,
    available_cpus,
    map_reduce,
)
from valuation.utils.algorithms import conjugate_gradient


def influences(
    model: SupervisedModel, data: Dataset, progress: bool = False, n_jobs: int = -1
) -> np.ndarray:
    """
    Calculates the influence of the training points j on the test points i, with matrix I_(ij). It does so by
    calculating the influence factors for all test points, with respect to the training points. Subsequently,
    all influence get calculated over the train set.

    :param model: A model which has to implement the TwiceDifferentiable interface in addition to SupervisedModel.
    :param data: A dataset, which hold training and test datasets.
    :param progress: whether to display progress bars.
    :param n_jobs: The number of jobs to use for processing
    :returns: A np.ndarray of size (N, M) where N is the number of training pointsand M is the number of test points.
    """

    # check if model support influence factors
    if not issubclass(model.__class__, TwiceDifferentiable):
        raise AttributeError(
            "Model is not twice differentiable, please implement interface."
        )

    # verify cpu correctness
    cpu_count = available_cpus()
    if n_jobs == -1:
        n_jobs = cpu_count
    elif n_jobs <= 0:
        raise AttributeError(
            "The number of jobs has to b bigger than zero or -1 for all available cores."
        )

    # ------------------------------------------------------------------------------------------------------------------

    twd: TwiceDifferentiable = model
    hvp = lambda v: twd.hvp(data.x_train, data.y_train, v, progress=progress)

    def _calculate_influence_factors(indices: np.ndarray, job_id: int) -> np.ndarray:
        """
        Calculates the influence factors, e.g. the inverse Hessian vector products if the test cores with respect
        to the Hessian induced by all training points.

        :param indices: A np.ndarray containing all indices of the test data which shall be evaluated in this run.
        :param job_id: A id which describes the current job id.
        :returns: A np.ndarray of size (N, D) containing the influence factors for each dimension and test sample.
        """
        c_x_test, c_y_test = data.x_test[indices], data.y_test[indices]
        test_grads = twd.grad(c_x_test, c_y_test, progress=progress)
        return conjugate_gradient(hvp, test_grads)[0]

    influence_factors_job = MapReduceJob.from_fun(
        _calculate_influence_factors, np.concatenate
    )
    influence_factors = map_reduce(
        influence_factors_job, np.arange(len(u.data.x_test)), num_jobs=n_jobs
    )[0]

    # ------------------------------------------------------------------------------------------------------------------

    def _calculate_influences(indices: np.ndarray, job_id: int) -> np.ndarray:
        """
        Calculates the influence from the influence factors and the scores of the training points.

        :param indices: A np.ndarray containing all indices of the training data which shall be evaluated in this run.
        :param job_id: A id which describes the current job id.
        :returns: A np.ndarray of size (N, K) containing the influence for each test sample and train sample.
        """
        c_x_train, c_y_train = data.x_train[indices], data.y_train[indices]
        train_grads = twd.grad(c_x_train, c_y_train)
        return contract("ta,va->tv", influence_factors, train_grads)

    influences_job = MapReduceJob.from_fun(
        _calculate_influences, functools.partial(np.concatenate, axis=1)
    )
    return map_reduce(influences_job, np.arange(len(data.x_train)), num_jobs=n_jobs)[0]
