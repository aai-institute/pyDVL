"""
Implementation of the algorithm footcite:t:`schoch_csshapley_2022`.
"""
import logging
import numbers
from concurrent.futures import FIRST_COMPLETED, wait
from copy import copy
from typing import cast

import numpy as np

from pydvl.utils import (
    ParallelConfig,
    Utility,
    effective_n_jobs,
    init_executor,
    init_parallel_backend,
)

__all__ = [
    "compute_classwise_shapley_values",
]

from tqdm import tqdm

from pydvl.utils.score import ClasswiseScorer
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.montecarlo import permutation_montecarlo_classwise_shapley
from pydvl.value.shapley.truncated import TruncationPolicy
from pydvl.value.stopping import MaxChecks, StoppingCriterion

logger = logging.getLogger(__name__)


def compute_classwise_shapley_values(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    normalize_values: bool = True,
    n_resample_complement_sets: int = 1,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
) -> ValuationResult:
    """
    Computes the classwise Shapley value by parallel processing. Independent workers
    are spawned to process the data in parallel. Once the data is aggregated, the values
    can be optionally normalized, depending on ``normalize_values``.

    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :param done: Function that checks whether the computation needs to stop.
    :param truncation: Callable function that decides whether to interrupt processing a
        permutation and set subsequent marginals to zero.
    :param normalize_values: Indicates whether to normalize the values by the variation
        in each class times their in-class accuracy.
    :param n_resample_complement_sets: Number of times to resample the complement set
        for each permutation.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
    :param min_elements_per_label: The minimum number of elements for each opposite
        label.
    :param n_jobs: Number of parallel jobs to run.
    :param config: Parallel configuration.
    :param progress: Whether to display progress bars for each job.
    :return: ValuationResult object containing computed data values.
    """

    _check_classwise_shapley_utility(u)

    parallel_backend = init_parallel_backend(config)
    u_ref = parallel_backend.put(u)
    # This represents the number of jobs that are running
    n_jobs = effective_n_jobs(n_jobs, config)
    # This determines the total number of submitted jobs
    # including the ones that are running
    n_submitted_jobs = 2 * n_jobs

    pbar = tqdm(disable=not progress, position=0, total=100, unit="%")
    accumulated_result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )
    terminate_exec = False
    with init_executor(max_workers=n_jobs, config=config) as executor:
        futures = set()
        # Initial batch of computations
        for _ in range(n_submitted_jobs):
            future = executor.submit(
                _classwise_shapley_one_step,
                u_ref,
                truncation=truncation,
                n_resample_complement_sets=n_resample_complement_sets,
                use_default_scorer_value=use_default_scorer_value,
                min_elements_per_label=min_elements_per_label,
            )
            futures.add(future)
        while futures:
            # Wait for the next futures to complete.
            completed_futures, futures = wait(
                futures, timeout=60, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                accumulated_result += future.result()
                if done(accumulated_result):
                    terminate_exec = True
                    break

            pbar.n = 100 * done.completion()
            pbar.refresh()
            if terminate_exec:
                break

            # Submit more computations
            # The goal is to always have `n_jobs`
            # computations running
            for _ in range(n_submitted_jobs - len(futures)):
                future = executor.submit(
                    _classwise_shapley_one_step,
                    u_ref,
                    truncation=truncation,
                    n_resample_complement_sets=n_resample_complement_sets,
                    use_default_scorer_value=use_default_scorer_value,
                    min_elements_per_label=min_elements_per_label,
                )
                futures.add(future)

    result = accumulated_result
    if normalize_values:
        result = _normalize_classwise_shapley_values(result, u)

    return result


def _classwise_shapley_one_step(
    u: Utility,
    *,
    truncation: TruncationPolicy,
    n_resample_complement_sets: int = 1,
    use_default_scorer_value: bool = True,
    min_elements_per_label: int = 1,
) -> ValuationResult:
    """Computes classwise Shapley value using truncated Monte Carlo permutation
    sampling for the subsets.

    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :param truncation: Callable function that decides whether to interrupt processing a
        permutation and set subsequent marginals to zero.
    :param n_resample_complement_sets: Number of times to resample the complement set
        for each permutation.
    :param use_default_scorer_value: Use default scorer value even if additional_indices
        is not None.
     :param min_elements_per_label: The minimum number of elements for each opposite
        label.
    :return: ValuationResult object containing computed data values.
    """
    result = ValuationResult.zeros(
        algorithm="classwise_shapley",
        indices=u.data.indices,
        data_names=u.data.data_names,
    )
    x_train, y_train = u.data.get_training_data(u.data.indices)
    unique_labels = np.unique(y_train)
    scorer = cast(ClasswiseScorer, copy(u.scorer))
    u.scorer = scorer

    for label in unique_labels:
        u.scorer.label = label
        result += permutation_montecarlo_classwise_shapley(
            u,
            label,
            done=MaxChecks(n_resample_complement_sets - 1),
            truncation=truncation,
            use_default_scorer_value=use_default_scorer_value,
            min_elements_per_label=min_elements_per_label,
        )

    return result


def _check_classwise_shapley_utility(u: Utility):
    """
    Verifies if the provided utility object supports classwise Shapley values.

    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    :raises: ValueError: If ``u.data`` is not a classification problem.
    :raises: ValueError: If ``u.scorer`` is not an instance of
        :class:`~pydvl.utils.score.ClassWiseScorer`
    """

    dim_correct = u.data.y_train.ndim == 1 and u.data.y_test.ndim == 1
    is_integral = all(
        map(
            lambda v: isinstance(v, numbers.Integral), (*u.data.y_train, *u.data.y_test)
        )
    )
    if not dim_correct or not is_integral:
        raise ValueError(
            "The supplied dataset has to be a 1-dimensional classification dataset."
        )

    if not isinstance(u.scorer, ClasswiseScorer):
        raise ValueError(
            "Please set a subclass of ClassWiseScorer object as scorer object of the"
            " utility. See scoring argument of Utility."
        )


def _normalize_classwise_shapley_values(
    result: ValuationResult,
    u: Utility,
) -> ValuationResult:
    """
    Normalize a valuation result specific to classwise Shapley.

    Each value corresponds to a class c and gets normalized by multiplying
    `in-class-score / sigma`. In this context `sigma` is the magnitude of all values
    belonging to the currently viewed class. See footcite:t:`schoch_csshapley_2022` for
    more details.

    :param result: ValuationResult object to be normalized.
    :param u: Utility object containing model, data, and scoring function. The scoring
        function should be of type :class:`~pydvl.utils.score.ClassWiseScorer`.
    """
    y_train = u.data.y_train
    unique_labels = np.unique(np.concatenate((y_train, u.data.y_test)))
    scorer = cast(ClasswiseScorer, u.scorer)

    for idx_label, label in enumerate(unique_labels):
        scorer.label = label
        active_elements = y_train == label
        indices_label_set = np.where(active_elements)[0]
        indices_label_set = u.data.indices[indices_label_set]

        u.model.fit(u.data.x_train, u.data.y_train)
        scorer.label = label
        in_cls_acc, _ = scorer.estimate_in_cls_and_out_of_cls_score(
            u.model, u.data.x_test, u.data.y_test
        )

        sigma = np.sum(result.values[indices_label_set])
        if sigma != 0:
            result.scale(indices_label_set, in_cls_acc / sigma)

    return result
