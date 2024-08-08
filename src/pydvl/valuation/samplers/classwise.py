from __future__ import annotations

from typing import Callable, Generator

import numpy as np
from numpy.typing import NDArray

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers.powerset import PowersetSampler
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import CSSample, IndexSetT, SampleGenerator
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["ClasswiseSampler", "get_unique_labels"]


def get_unique_labels(array: NDArray) -> NDArray:
    """Labels of the dataset."""
    # Object, String, Unicode, Unsigned integer, Signed integer, boolean
    if array.dtype.kind in "OSUiub":
        return np.unique(array)
    raise ValueError("Dataset must be categorical to have unique labels.")


class ClasswiseSampler(StochasticSamplerMixin, IndexSampler):
    def __init__(
        self,
        in_class: IndexSampler,
        out_of_class: PowersetSampler,
        *,
        min_elements_per_label: int = 1,
        max_repeated_ooc_sampling: int = 100,
    ):
        super().__init__()
        self.in_class = in_class
        self.out_of_class = out_of_class
        self.min_elements_per_label = min_elements_per_label
        self.max_repeated_ooc_sampling = max_repeated_ooc_sampling

    def interrupt(self):
        self.in_class.interrupt()
        self.out_of_class.interrupt()

    def from_data(self, data: Dataset) -> Generator[list[CSSample], None, None]:
        """
        assert self.label is not None

        without_label = np.where(data.y != self.label)[0]
        with_label = np.where(data.y == self.label)[0]

        # HACK: the outer sampler is over full subsets of T_{-y_i}
        # self.out_of_class._index_iteration = NoIndexIteration
        """

        labels = get_unique_labels(data.y)
        n_labels = len(labels)
        while not self._interrupted:
            batch: list[CSSample] = []

            label = self._rng.choice(labels)
            with_label = np.where(data.y == label)[0]
            without_label = np.where(data.y != label)[0]
            # NOTE: The inner sampler can be a permutation sampler => we need to
            #  return batches of the same size as that sampler in order for the
            #  in_class strategy to work correctly.
            # i.e. len(batch) == len(ic_batch)
            for _ in range(self.in_class.batch_size):
                if self.min_elements_per_label <= 0:
                    ooc_sample = next(self.out_of_class._generate(without_label))
                else:
                    # Using a high number of iterations
                    # instead of a while True loop to avoid
                    # having an accidental infinite loop
                    for i in range(self.max_repeated_ooc_sampling):
                        i += 1
                        # We make sure that we have at least
                        # `min_elements_per_label` elements per label per sample
                        ooc_sample = next(self.out_of_class._generate(without_label))
                        n_unique_sample_labels = len(
                            get_unique_labels(data.y[ooc_sample.subset])
                        )
                        if n_unique_sample_labels == n_labels - 1:
                            break
                    else:
                        raise RuntimeError(
                            f"Could not find out-of-class sample with at least"
                            f" {self.min_elements_per_label} elements per sample after retrying"
                            f" {self.max_repeated_ooc_sampling} times."
                            f" Consider increasint max_repeated_ooc_sampling"
                        )

                ic_sample = next(self.in_class._generate(with_label))
                batch.append(
                    CSSample(
                        idx=ic_sample.idx,
                        label=label,
                        subset=ic_sample.subset,
                        ooc_subset=ooc_sample.subset,
                    )
                )

            yield batch

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        raise AttributeError("Cannot sample from indices directly.")

    @staticmethod
    def weight(n: int, subset_len: int) -> float:
        raise AttributeError("The weight should come from the in_class sampler")

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy[IndexSampler]:
        return self.in_class.make_strategy(utility, coefficient)
