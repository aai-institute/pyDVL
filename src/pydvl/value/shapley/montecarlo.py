r"""
Monte Carlo approximations to Shapley Data values.

.. warning::
   You probably want to use the common interface provided by
   :func:`~pydvl.value.shapley.compute_shapley_values` instead of directly using
   the functions in this module.

Because exact computation of Shapley values requires $\mathcal{O}(2^n)$
re-trainings of the model, several Monte Carlo approximations are available.
The first two sample from the powerset of the training data directly:
:func:`combinatorial_montecarlo_shapley` and :func:`owen_sampling_shapley`. The
latter uses a reformulation in terms of a continuous extension of the utility.

Alternatively, employing another reformulation of the expression above as a sum
over permutations, one has the implementation in
:func:`permutation_montecarlo_shapley`, or using an early stopping strategy to
reduce computation :func:`truncated_montecarlo_shapley`.

.. seealso::
   It is also possible to use :func:`~pydvl.value.shapley.gt.group_testing_shapley`
   to reduce the number of evaluations of the utility. The method is however
   typically outperformed by others in this module.

.. seealso::
   Additionally, you can consider grouping your data points using
   :class:`~pydvl.utils.dataset.GroupedDataset` and computing the values of the
   groups instead. This is not to be confused with "group testing" as
   implemented in :func:`~pydvl.value.shapley.gt.group_testing_shapley`: any of
   the algorithms mentioned above, including Group Testing, can work to valuate
   groups of samples as units.
"""
import logging
import math
import operator
from functools import reduce
from itertools import cycle, takewhile
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pydvl.utils import Dataset, Utility, random_powerset_group_conditional
from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob
from pydvl.utils.utility import Utility
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import NoTruncation, TruncationPolicy
from pydvl.value.stopping import StoppingCriterion

logger = logging.getLogger(__name__)

__all__ = [
    "permutation_montecarlo_shapley",
    "permutation_montecarlo_classwise_shapley",
    "combinatorial_montecarlo_shapley",
]


def _permutation_montecarlo_shapley(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    algorithm_name: str = "permutation_montecarlo_shapley",
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function for :func:`permutation_montecarlo_shapley`.

    Computes marginal utilities of each training sample in
    :obj:`pydvl.utils.utility.Utility.data` by iterating through randomly
    sampled permutations.

    :param u: Utility object with model, data, and scoring function
    :param done: Check on the results which decides when to stop
    :param truncation: A callable which decides whether to interrupt
        processing a permutation and set all subsequent marginals to zero.
    :param algorithm_name: For the results object. Used internally by different
        variants of Shapley using this subroutine
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress (e.g. to place progres bars)
    :return: An object with the results
    """
    result = ValuationResult.zeros(
        algorithm=algorithm_name, indices=u.data.indices, data_names=u.data.data_names
    )

    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    while not done(result):
        pbar.n = 100 * done.completion()
        pbar.refresh()
        permutation = np.random.permutation(u.data.indices)
        result += _permutation_montecarlo_shapley_rollout(
            u, permutation, truncation=truncation, algorithm_name=algorithm_name
        )

    return result


def permutation_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    truncation: TruncationPolicy = NoTruncation(),
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes an approximate Shapley value by sampling independent index
    permutations to approximate the sum:

    $$
    v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
    \tilde{w}( | \sigma_{:i} | )[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})],
    $$

    where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
    position where $i$ appears (see :ref:`data valuation` for details).

    :param u: Utility object with model, data, and scoring function.
    :param done: function checking whether computation must stop.
    :param truncation: An optional callable which decides whether to
        interrupt processing a permutation and set all subsequent marginals to
        zero. Typically used to stop computation when the marginal is small.
    :param n_jobs: number of jobs across which to distribute the computation.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """

    map_reduce_job: MapReduceJob[Utility, ValuationResult] = MapReduceJob(
        u,
        map_func=_permutation_montecarlo_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(
            algorithm_name="permutation_montecarlo_shapley",
            done=done,
            truncation=truncation,
            progress=progress,
        ),
        config=config,
        n_jobs=n_jobs,
    )
    return map_reduce_job()


def permutation_montecarlo_classwise_shapley(
    u: Utility,
    label: int,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
) -> ValuationResult:
    """
    Samples a random subset of the complement set and computes the truncated Monte Carlo
    estimator.

    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :param done: Function checking whether computation needs to stop.
    :param label: The label for which to sample the complement (e.g. all other labels)
    :param truncation: Callable which decides whether to interrupt processing a
        permutation and set all subsequent marginals to zero.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
    :param min_elements_per_label: The minimum number of elements for each opposite
        label.
    :return: ValuationResult object containing computed data values.
    """

    algorithm_name = "classwise_shapley"
    result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    _, y_train = u.data.get_training_data(u.data.indices)
    class_indices_set, class_complement_indices_set = split_indices_by_label(
        u.data.indices,
        y_train,
        label,
    )
    _, complement_y_train = u.data.get_training_data(class_complement_indices_set)
    indices_permutation = np.random.permutation(class_indices_set)

    for subset_idx, subset_complement in enumerate(
        random_powerset_group_conditional(
            class_complement_indices_set,
            complement_y_train,
            min_elements_per_group=min_elements_per_label,
        )
    ):
        result += _permutation_montecarlo_shapley_rollout(
            u,
            indices_permutation,
            additional_indices=subset_complement,
            truncation=truncation,
            algorithm_name=algorithm_name,
            use_default_scorer_value=use_default_scorer_value,
        )
        if done(result):
            break

    return result


def _permutation_montecarlo_shapley_rollout(
    u: Utility,
    permutation: NDArray[np.int_],
    *,
    truncation: TruncationPolicy,
    algorithm_name: str,
    additional_indices: Optional[NDArray[np.int_]] = None,
    use_default_scorer_value: bool = True,
) -> ValuationResult:
    """
    A truncated version of a permutation-based MC estimator for classwise Shapley
    values. It generates a permutation p[i] of the class label indices and iterates over
    all subsets starting from the empty set to the full set of indices.

    :param u: Utility object containing model, data, and scoring function. The scoring
        function should to be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :param permutation: Permutation of indices to be considered.
    :param truncation: Callable which decides whether to interrupt processing a
        permutation and set all subsequent marginals to zero.
    :param additional_indices: Set of additional indices for data points which should be
        always considered.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
    :return: ValuationResult object containing computed data values.
    """
    if (
        additional_indices is not None
        and len(np.intersect1d(permutation, additional_indices)) > 0
    ):
        raise ValueError(
            "The class label set and the complement set have to be disjoint."
        )

    result = ValuationResult.zeros(
        algorithm=algorithm_name,
        indices=u.data.indices,
        data_names=u.data.data_names,
    )

    prev_score = (
        u.default_score
        if (
            use_default_scorer_value
            or additional_indices is None
            or additional_indices is not None
            and len(additional_indices) == 0
        )
        else u(additional_indices)
    )

    truncation_u = u
    if additional_indices is not None:
        # hack to calculate the correct value in reset.
        truncation_indices = np.sort(np.concatenate((permutation, additional_indices)))
        truncation_u = Utility(
            u.model,
            Dataset(
                u.data.x_train[truncation_indices],
                u.data.y_train[truncation_indices],
                u.data.x_test,
                u.data.y_test,
            ),
            u.scorer,
        )
    truncation.reset(truncation_u)

    is_terminated = False
    for i, idx in enumerate(permutation):
        if is_terminated or (is_terminated := truncation(i, prev_score)):
            score = prev_score
        else:
            score = u(
                np.concatenate((permutation[: i + 1], additional_indices))
                if additional_indices is not None and len(additional_indices) > 0
                else permutation[: i + 1]
            )

        marginal = score - prev_score
        result.update(idx, marginal)
        prev_score = score

    return result


def _combinatorial_montecarlo_shapley(
    indices: Sequence[int],
    u: Utility,
    done: StoppingCriterion,
    *,
    progress: bool = False,
    job_id: int = 1,
) -> ValuationResult:
    """Helper function for :func:`combinatorial_montecarlo_shapley`.

    This is the code that is sent to workers to compute values using the
    combinatorial definition.

    :param indices: Indices of the samples to compute values for.
    :param u: Utility object with model, data, and scoring function
    :param done: Check on the results which decides when to stop sampling
        subsets for an index.
    :param progress: Whether to display progress bars for each job.
    :param job_id: id to use for reporting progress
    :return: A tuple of ndarrays with estimated values and standard errors
    """
    n = len(u.data)

    # Correction coming from Monte Carlo integration so that the mean of the
    # marginals converges to the value: the uniform distribution over the
    # powerset of a set with n-1 elements has mass 2^{n-1} over each subset. The
    # additional factor n corresponds to the one in the Shapley definition
    correction = 2 ** (n - 1) / n
    result = ValuationResult.zeros(
        algorithm="combinatorial_montecarlo_shapley",
        indices=np.array(indices, dtype=np.int_),
        data_names=[u.data.data_names[i] for i in indices],
    )

    repeat_indices = takewhile(lambda _: not done(result), cycle(indices))
    pbar = tqdm(disable=not progress, position=job_id, total=100, unit="%")
    for idx in repeat_indices:
        pbar.n = 100 * done.completion()
        pbar.refresh()
        # Randomly sample subsets of full dataset without idx
        subset = np.setxor1d(u.data.indices, [idx], assume_unique=True)
        s = next(random_powerset(subset, n_samples=1))
        marginal = (u({idx}.union(s)) - u(s)) / math.comb(n - 1, len(s))
        result.update(idx, correction * marginal)

    return result


def combinatorial_montecarlo_shapley(
    u: Utility,
    done: StoppingCriterion,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    r"""Computes an approximate Shapley value using the combinatorial
    definition:

    $$v_u(i) = \frac{1}{n} \sum_{S \subseteq N \setminus \{i\}}
    \binom{n-1}{ | S | }^{-1} [u(S \cup \{i\}) − u(S)]$$

    This consists of randomly sampling subsets of the power set of the training
    indices in :attr:`~pydvl.utils.utility.Utility.data`, and computing their
    marginal utilities. See :ref:`data valuation` for details.

    Note that because sampling is done with replacement, the approximation is
    poor even for $2^{m}$ subsets with $m>n$, even though there are $2^{n-1}$
    subsets for each $i$. Prefer
    :func:`~pydvl.shapley.montecarlo.permutation_montecarlo_shapley`.

    Parallelization is done by splitting the set of indices across processes and
    computing the sum over subsets $S \subseteq N \setminus \{i\}$ separately.

    :param u: Utility object with model, data, and scoring function
    :param done: Stopping criterion for the computation.
    :param n_jobs: number of parallel jobs across which to distribute the
        computation. Each worker receives a chunk of
        :attr:`~pydvl.utils.dataset.Dataset.indices`
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param progress: Whether to display progress bars for each job.
    :return: Object with the data values.
    """

    map_reduce_job: MapReduceJob[NDArray, ValuationResult] = MapReduceJob(
        u.data.indices,
        map_func=_combinatorial_montecarlo_shapley,
        reduce_func=lambda results: reduce(operator.add, results),
        map_kwargs=dict(u=u, done=done, progress=progress),
        n_jobs=n_jobs,
        config=config,
    )
    return map_reduce_job()


def split_indices_by_label(
    indices: NDArray[np.int_], labels: NDArray[np.int_], label: int
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Splits the indices into two sets based on the value of  ``label``: those samples
    with and without that label.

    :param indices: The indices to be used for referring to the data.
    :param labels: Corresponding labels for the indices.
    :param label: Label to be used for splitting.
    :return: Tuple with two sets of indices.
    """
    active_elements = labels == label
    class_indices_set = np.where(active_elements)[0]
    class_complement_indices_set = np.where(~active_elements)[0]
    class_indices_set = indices[class_indices_set]
    class_complement_indices_set = indices[class_complement_indices_set]
    return class_indices_set, class_complement_indices_set
