r"""
Class-wise Shapley (Schoch et al., 2022)[^1] offers a Shapley framework tailored
for classification problems. Let $D$ be a dataset, $D_{y_i}$ be the subset of
$D$ with labels $y_i$, and $D_{-y_i}$ be the complement of $D_{y_i}$ in $D$. The
key idea is that a sample $(x_i, y_i)$, might enhance the overall performance on
$D$, while being detrimental for the performance on $D_{y_i}$. The Class-wise
value is defined as:

$$
v_u(i) = \frac{1}{2^{|D_{-y_i}|}} \sum_{S_{-y_i}} \frac{1}{|D_{y_i}|!}
\sum_{S_{y_i}} \binom{|D_{y_i}|-1}{|S_{y_i}|}^{-1}
[u( S_{y_i} \cup \{i\} | S_{-y_i} ) − u( S_{y_i} | S_{-y_i})],
$$

where $S_{y_i} \subseteq D_{y_i} \setminus \{i\}$ and $S_{-y_i} \subseteq
D_{-y_i}$.

!!! tip "Analysis of Class-wise Shapley"
    For a detailed analysis of the method, with comparison to other valuation
    techniques, please refer to the [main documentation][class-wise-shapley].

In practice, the quantity above is estimated using Monte Carlo sampling of
the powerset and the set of index permutations. This results in the estimator

$$
v_u(i) = \frac{1}{K} \sum_k \frac{1}{L} \sum_l
[u(\sigma^{(l)}_{:i} \cup \{i\} | S^{(k)} ) − u( \sigma^{(l)}_{:i} | S^{(k)})],
$$

with $S^{(1)}, \dots, S^{(K)} \subseteq T_{-y_i},$ $\sigma^{(1)}, \dots,
\sigma^{(L)} \in \Pi(T_{y_i}\setminus\{i\}),$ and $\sigma^{(l)}_{:i}$ denoting
the set of indices in permutation $\sigma^{(l)}$ before the position where $i$
appears. The sets $T_{y_i}$ and $T_{-y_i}$ are the training sets for the labels
$y_i$ and $-y_i$, respectively.

??? info "Notes for derivation of test cases"
    The unit tests include the following manually constructed data:
    Let $D=\{(1,0),(2,0),(3,0),(4,1)\}$ be the test set and $T=\{(1,0),(2,0),(3,1),(4,1)\}$
    the train set. This specific dataset is chosen as it allows to solve the model

    $$y = \max(0, \min(1, \text{round}(\beta^T x)))$$

    in closed form $\beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}$. From the closed-form
    solution, the tables for in-class accuracy $a_S(D_{y_i})$ and out-of-class accuracy 
    $a_S(D_{-y_i})$ can be calculated. By using these tables and setting 
    $\{S^{(1)}, \dots, S^{(K)}\} = 2^{T_{-y_i}}$ and 
    $\{\sigma^{(1)}, \dots, \sigma^{(L)}\} = \Pi(T_{y_i}\setminus\{i\})$,
    the Monte Carlo estimator can be evaluated ($2^M$ is the powerset of $M$).
    The details of the derivation are left to the eager reader.

# References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.
"""

from __future__ import annotations

import logging
from typing import Callable, Generator

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from pydvl.utils import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler, PowersetSampler
from pydvl.valuation.samplers.base import EvaluationStrategy
from pydvl.valuation.samplers.powerset import NoIndexIteration
from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import BatchGenerator, IndexSetT
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utility.classwise import CSSample
from pydvl.valuation.utils import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)

__all__ = ["ClasswiseShapley"]

logger = logging.getLogger(__name__)


def unique_labels(array: NDArray) -> NDArray:
    """Labels of the dataset."""
    # Object, String, Unicode, Unsigned integer, Signed integer, boolean
    if array.dtype.kind in "OSUiub":
        return np.unique(array)
    raise ValueError("Dataset must be categorical to have unique labels.")


class ClasswiseSampler(IndexSampler):
    def __init__(
        self,
        in_class: IndexSampler,
        out_of_class: PowersetSampler,
        label: int | None = None,
    ):
        super().__init__()
        self.in_class = in_class
        self.out_of_class = out_of_class
        self.label = label

    def for_label(self, label: int) -> ClasswiseSampler:
        return ClasswiseSampler(self.in_class, self.out_of_class, label)

    def from_data(self, data: Dataset) -> Generator[list[CSSample], None, None]:
        assert self.label is not None

        without_label = np.where(data.y != self.label)[0]
        with_label = np.where(data.y == self.label)[0]

        # HACK: the outer sampler is over full subsets of T_{-y_i}
        self.out_of_class._index_iteration = NoIndexIteration

        for ooc_batch in self.out_of_class.from_indices(without_label):
            # NOTE: The inner sampler can be a permutation sampler => we need to
            #  return batches of the same size as that sampler in order for the
            #  in_class strategy to work correctly.
            for ooc_sample in ooc_batch:
                for ic_batch in self.in_class.from_indices(with_label):
                    # FIXME? this sends the same out_of_class_subset for all samples
                    #   maybe a few 10s of KB... probably irrelevant
                    yield [
                        CSSample(
                            idx=ic_sample.idx,
                            label=self.label,
                            subset=ooc_sample.subset,
                            in_class_subset=ic_sample.subset,
                        )
                        for ic_sample in ic_batch
                    ]

    def from_indices(self, indices: IndexSetT) -> BatchGenerator:
        raise AttributeError("Cannot sample from indices directly.")

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy[IndexSampler]:
        return self.in_class.make_strategy(utility, coefficient)


class ClasswiseShapley(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: ClasswiseSampler,
        is_done: StoppingCriterion,
        progress: bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.labels: NDArray | None = None
        if not isinstance(utility.scorer, ClasswiseSupervisedScorer):
            raise ValueError("Scorer must be a ClasswiseScorer.")
        self.scorer: ClasswiseSupervisedScorer = utility.scorer
        self.is_done = is_done
        self.progress = progress

    def fit(self, data: Dataset):
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations
            algorithm=f"classwise-shapley",
            indices=data.indices,
            data_names=data.data_names,
        )
        ensure_backend_has_generator_return()

        parallel = Parallel(return_as="generator_unordered")

        self.utility.training_data = data
        self.labels = unique_labels(data.y)

        with make_parallel_flag() as flag:
            # FIXME, DUH: this loop needs to be in the sampler or we will never converge
            for label in self.labels:
                sampler = self.sampler.for_label(label)
                strategy = sampler.make_strategy(self.utility)
                processor = delayed(strategy.process)
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in sampler.from_data(data)
                )
                for evaluation in Progress(delayed_evals, self.is_done):
                    self.result.update(evaluation.idx, evaluation.update)
                    if self.is_done(self.result):
                        flag.set()
                        break

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
        assert self.result is not None
        assert self.utility.training_data is not None

        u = self.utility

        logger.info("Normalizing valuation result.")
        u.model.fit(u.training_data.x, u.training_data.y)

        for idx_label, label in enumerate(self.labels):
            self.scorer.label = label
            active_elements = u.training_data.y == label
            indices_label_set = np.where(active_elements)[0]
            indices_label_set = u.training_data.indices[indices_label_set]

            self.scorer.label = label
            in_class_acc, _ = self.scorer.compute_in_and_out_of_class_scores(u.model)

            sigma = np.sum(self.result.values[indices_label_set])
            if sigma != 0:
                self.result.scale(in_class_acc / sigma, indices=indices_label_set)

        return self.result
