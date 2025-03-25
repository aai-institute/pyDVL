r"""
Class-wise Shapley (Schoch et al., 2022)[^1] is a semi-value tailored for classification
problems.

The core intuition behind the method is that a sample might enhance the overall
performance of the model, while being detrimental for the performance when the model
is restricted to items of the same class, and vice versa.

!!! tip "Analysis of Class-wise Shapley"
    For a detailed explanation and analysis of the method, with comparison to other
    valuation techniques, please refer to the [main documentation][classwise-shapley-intro]
    and to Semmler and de Benito Delgado (2024).[^2]

## References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.
[^2]: <a name="semmler_re_2024"></a>Semmler, Markus, and Miguel de Benito Delgado.
    [[Re] Classwise-Shapley Values for Data
    Valuation](https://openreview.net/forum?id=srFEYJkqD7&noteId=zVi6DINuXT).
    Transactions on Machine Learning Research, July 2024.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from pydvl.utils.progress import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.parallel import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.classwise import ClasswiseSampler, get_unique_labels
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.classwise import ClasswiseModelUtility

__all__ = ["ClasswiseShapleyValuation"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ClasswiseShapleyValuation(Valuation):
    """Class to compute Class-wise Shapley values.

    Args:
        utility: Class-wise utility object with model and class-wise scoring function.
        sampler: Class-wise sampling scheme to use.
        is_done: Stopping criterion to use.
        progress: Whether to show a progress bar.
        normalize_values: Whether to normalize values after valuation.
    """

    algorithm_name = "Classwise-Shapley"

    def __init__(
        self,
        utility: ClasswiseModelUtility,
        sampler: ClasswiseSampler,
        is_done: StoppingCriterion,
        progress: dict[str, Any] | bool = False,
        *,
        normalize_values: bool = True,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.labels: NDArray | None = None
        if not isinstance(utility.scorer, ClasswiseSupervisedScorer):
            raise ValueError("scorer must be an instance of ClasswiseSupervisedScorer")
        self.scorer: ClasswiseSupervisedScorer = utility.scorer
        self.is_done = is_done
        self.tqdm_args: dict[str, Any] = {
            "desc": f"{self.__class__.__name__}: {str(is_done)}"
        }
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        else:
            self.tqdm_args.update(progress if isinstance(progress, dict) else {})
        self.normalize_values = normalize_values

    def fit(self, data: Dataset):
        # TODO?
        if isinstance(data, GroupedDataset):
            raise ValueError(
                "GroupedDataset is not supported for ClasswiseShapleyValuation"
            )

        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"{self.__class__.__name__}-{self.utility.__class__.__name__}-{self.sampler.__class__.__name__}-{self.is_done}",
            indices=data.indices,
            data_names=data.names,
        )
        ensure_backend_has_generator_return()

        self.is_done.reset()
        self.utility = self.utility.with_dataset(data)

        strategy = self.sampler.make_strategy(self.utility, None)
        updater = self.sampler.result_updater(self.result)
        processor = delayed(strategy.process)

        sample_generator = self.sampler.from_data(data)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in sample_generator
                )

                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for evaluation in batch:
                        self.result = updater.process(evaluation)
                    if self.is_done(self.result):
                        flag.set()
                        self.sampler.interrupt()
                        break

        if self.normalize_values:
            self._normalize()

        return self

    def _normalize(self) -> ValuationResult:
        r"""
        Normalize a valuation result specific to classwise Shapley.

        Each value $v_i$ associated with the sample $(x_i, y_i)$ is normalized by
        multiplying it with $a_S(D_{y_i})$ and dividing by $\sum_{j \in D_{y_i}} v_j$.
        For more details, see (Schoch et al., 2022) <sup><a
        href="#schoch_csshapley_2022">1</a> </sup>.

        Returns:
            Normalized ValuationResult object.
        """
        if self.result is None:
            raise ValueError("You must call fit before calling _normalize()")

        if self.utility.training_data is None:
            raise ValueError("You should call fit before calling _normalize()")

        logger.info("Normalizing valuation result.")
        x, y = self.utility.training_data.data()
        unique_labels = get_unique_labels(y)
        self.utility.model.fit(x, y)

        for idx_label, label in enumerate(unique_labels):
            active_elements = y == label
            indices_label_set = np.where(active_elements)[0]
            indices_label_set = self.utility.training_data.indices[indices_label_set]

            self.scorer.label = label
            in_class_acc, _ = self.scorer.compute_in_and_out_of_class_scores(
                self.utility.model
            )

            sigma = np.sum(self.result.values[indices_label_set])
            if sigma != 0:
                self.result.scale(in_class_acc / sigma, data_indices=indices_label_set)

        return self.result
