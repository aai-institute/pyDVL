"""
Class-wise sampler for the [class-wise Shapley][intro-to-cw-shapley] valuation method.

The class-wise Shapley method, introduced by Schoch et al., 2022[^1], uses a so-called
*set-conditional marginal Shapley value* that requires selectively sampling subsets of
data points with the same or a different class from that of the data point of interest.

This sampling scheme is divided into an outer and an inner sampler. The outer one is any
subclass of [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler] that
generates subsets of the complement set of the data point of interest. The inner sampler
is any subclass of [IndexSampler][pydvl.valuation.samplers.base.IndexSampler], typically
(and in the paper) a
[PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler].

## References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.

"""

from __future__ import annotations

from itertools import cycle, islice
from typing import Callable, Generator, Iterable, Mapping, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers.powerset import NoIndexIteration, PowersetSampler
from pydvl.valuation.types import (
    ClasswiseSample,
    IndexSetT,
    SampleGenerator,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["ClasswiseSampler", "get_unique_labels"]

U = TypeVar("U")
V = TypeVar("V")


def roundrobin(
    batch_generators: Mapping[U, Iterable[V]],
) -> Generator[tuple[U, V], None, None]:
    """Take samples from batch generators in order until all of them are exhausted.

    This was heavily inspired by the roundrobin recipe
    in the official Python documentation for the itertools package.

    Examples:
        >>> from pydvl.valuation.samplers.classwise import roundrobin
        >>> list(roundrobin({"A": "123"}, {"B": "456"}))
        [("A", "1"), ("B", "4"), ("A", "2"), ("B", "5"), ("A", "3"), ("B", "6")]

    Args:
        batch_generators: dictionary mapping labels to batch generators.

    Returns:
        Combined generators
    """
    n_active = len(batch_generators)
    remaining_generators = cycle(
        (label, iter(it).__next__) for label, it in batch_generators.items()
    )
    while n_active:
        try:
            for label, next_generator in remaining_generators:
                yield label, next_generator()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            n_active -= 1
            remaining_generators = cycle(islice(remaining_generators, n_active))


def get_unique_labels(array: NDArray) -> NDArray:
    """Returns unique labels in a categorical dataset.

    Args:
        array: The input array to find unique labels from. It should be of
               categorical types such as Object, String, Unicode, Unsigned
               integer, Signed integer, or Boolean.

    Returns:
        An array of unique labels.

    Raises:
        ValueError: If the input array is not of a categorical type.
    """
    # Object, String, Unicode, Unsigned integer, Signed integer, boolean
    if array.dtype.kind in "OSUiub":
        return cast(NDArray, np.unique(array))
    raise ValueError(
        f"Input array has an unsupported data type for categorical labels: {array.dtype}. "
        "Expected types: Object, String, Unicode, Unsigned integer, Signed integer, or Boolean."
    )


class ClasswiseSampler(IndexSampler):
    """A sampler that samples elements from a dataset in two steps, based on the labels.

    Used by the [class-wise Shapley valuation
    method][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation].

    Args:
        in_class: Sampling scheme for elements of a given label.
        out_of_class: Sampling scheme for elements of different labels,
            i.e., the complement set.
        min_elements_per_label: Minimum number of elements per label
            to sample from the complement set, i.e., out of class elements.
    """

    def __init__(
        self,
        in_class: IndexSampler,
        out_of_class: PowersetSampler,
        *,
        min_elements_per_label: int = 1,
        batch_size: int = 1,
    ):
        super().__init__(batch_size=batch_size)
        self.in_class = in_class
        self.out_of_class = out_of_class
        self.min_elements_per_label = min_elements_per_label

    def interrupt(self) -> None:
        """Interrupts the current sampler as well as the passed in samplers"""
        super().interrupt()
        self.in_class.interrupt()
        self.out_of_class.interrupt()

    def from_data(self, data: Dataset) -> Generator[list[ClasswiseSample], None, None]:
        labels = get_unique_labels(data.data().y)
        n_labels = len(labels)

        # HACK: the outer sampler is over full subsets of T_{-y_i}
        # By default, powerset samplers remove the index from the generated
        # subset but in this case we want all indices.
        # The index for which we compute the value will be removed by
        # the in_class sampler instead.
        if not issubclass(self.out_of_class._index_iterator_cls, NoIndexIteration):
            self.out_of_class._index_iterator_cls = NoIndexIteration

        out_of_class_batch_generators = {}

        for label in labels:
            without_label = np.where(data.data().y != label)[0]
            out_of_class_batch_generators[label] = self.out_of_class.generate_batches(
                without_label
            )

        for label, ooc_batch in roundrobin(out_of_class_batch_generators):
            for ooc_sample in ooc_batch:
                if self.min_elements_per_label > 0:
                    # We make sure that we have at least
                    # `min_elements_per_label` elements per label per sample
                    n_unique_sample_labels = len(
                        get_unique_labels(data.data().y[ooc_sample.subset])
                    )
                    if n_unique_sample_labels < n_labels - 1:
                        continue

                with_label = np.where(data.data().y == label)[0]
                for ic_batch in self.in_class.generate_batches(with_label):
                    batch: list[ClasswiseSample] = []
                    for ic_sample in ic_batch:
                        batch.append(
                            ClasswiseSample(
                                idx=ic_sample.idx,
                                label=label,
                                subset=ic_sample.subset,
                                ooc_subset=ooc_sample.subset,
                            )
                        )
                    self._n_samples += len(batch)
                    yield batch

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        # This is not needed because this sampler is used
        # by calling the `from_data` method instead of the `generate_batches` method.
        raise AttributeError("Cannot sample from indices directly.")

    def log_weight(self, n: int, subset_len: int) -> float:
        # CW-Shapley uses the evaluation strategy from the in-class sampler, so this
        # method should never be called.
        raise AttributeError("The weight should come from the in-class sampler")

    def sample_limit(self, indices: IndexSetT) -> int | None:
        # The sample list cannot be computed without accessing the label
        # information and using that to compute the sample limits
        # of the in-class and out-of-class samplers first.
        raise AttributeError(
            "The sample limit cannot be computed without the label information."
        )

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy[IndexSampler, ValueUpdate]:
        return self.in_class.make_strategy(utility, log_coefficient)
