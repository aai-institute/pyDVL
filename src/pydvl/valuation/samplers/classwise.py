"""
Class-wise sampler for the [class-wise Shapley][classwise-shapley-intro] valuation method.

The class-wise Shapley method, introduced by Schoch et al., 2022[^1], uses a so-called
*set-conditional marginal Shapley value* that requires selectively sampling subsets of
data points with the same or a different class from that of the data point of interest.

This sampling scheme is divided into an outer and an inner sampler.

* The **outer sampler** is any subclass of
  [PowersetSampler][pydvl.valuation.samplers.powerset.PowersetSampler] that generates
  subsets within the complement set of the data point of interest, and _with a different
  label_ (so-called "out-of-class" samples, denoted by $S_{-y_i}$ in this documentation).
* The **inner sampler** is any subclass of
  [IndexSampler][pydvl.valuation.samplers.base.IndexSampler], typically (and in the
  paper) a
  [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]. It
  returns so-called "in-class" samples (denoted by $S_{y_i}$ in this documentation) from
  the set $N_{y_i}$, i.e., the set of all indices with the same label as the data point
  of interest.
    !!! info "Restricting the number of inner samples"
        Because of the nested sampling procedure, it is necessary to limit the amount
        of in-class samples to a finite number. This is done by setting the
        `max_in_class_samples` parameter. For finite samplers, it can be left as `None`
        to let them run until completion.

!!! info
    For more information on the class-wise Shapley method, as well as a summary of the
    reproduction results by Semmler and de Benito Delgado (2024)[^2] see [the main
    documentation][classwise-shapley-intro] for the method.

## References

[^1]: <a name="schoch_csshapley_2022"></a>Schoch, Stephanie, Haifeng Xu, and
    Yangfeng Ji. [CS-Shapley: Class-wise Shapley Values for Data Valuation in
    Classification](https://openreview.net/forum?id=KTOcrOR5mQ9). In Proc. of
    the Thirty-Sixth Conference on Neural Information Processing Systems
    (NeurIPS). New Orleans, Louisiana, USA, 2022.
[^2]: <a name="semmler_classwise_2024"></a>Semmler, Markus, and Miguel de Benito
    Delgado. [[Re] Classwise-Shapley Values for Data
    Valuation](https://openreview.net/forum?id=srFEYJkqD7&noteId=zVi6DINuXT).
    Transactions on Machine Learning Research, July 2024.

"""

from __future__ import annotations

from itertools import cycle, islice
from typing import Generator, Iterable, Mapping, TypeVar, cast

import numpy as np
from more_itertools import chunked, flatten

from pydvl.utils.array import Array, array_unique, is_categorical, try_torch_import
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers.base import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers.powerset import NoIndexIteration, PowersetSampler
from pydvl.valuation.types import (
    BatchGenerator,
    ClasswiseSample,
    IndexSetT,
    SampleGenerator,
    SemivalueCoefficient,
    ValueUpdate,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["ClasswiseSampler"]

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


def get_unique_labels(arr: Array[int]) -> Array[int]:
    """Returns unique labels in a categorical dataset.

    Args:
        arr: The input array to find unique labels from. It should be of
             categorical types such as Object, String, Unicode, Unsigned
             integer, Signed integer, or Boolean.

    Returns:
        An array of unique labels.

    Raises:
        ValueError: If the input array is not of a categorical type.
    """
    if is_categorical(arr):
        return array_unique(arr)
    else:
        raise ValueError(
            f"Input array has an unsupported data type for categorical labels: {type(arr)}. "
            "Expected types: Object, String, Unicode, Unsigned integer, Signed integer, or Boolean."
        )


class ClasswiseSampler(IndexSampler[ClasswiseSample, ValueUpdate]):
    """A sampler that samples elements from a dataset in two steps, based on the labels.

    It proceeds by sampling out-of-class indices (training points with a different
    label to the point of interest), and in-class indices (training points with the
    same label as the point of interest).

    Used by the [class-wise Shapley valuation
    method][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation].

    Args:
        in_class: Sampling scheme for elements of a given label (inner sampler).
            Typically, a
            [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler].
        out_of_class: Sampling scheme for elements of different labels (outer sampler).
            E.g. a [UniformSampler][pydvl.valuation.samplers.uniform.UniformSampler] or
            a [VRDSSampler][pydvl.valuation.samplers.stratified.VRDSSampler]. This
            sampler must use
            [NoIndexIteration][pydvl.valuation.samplers.powerset.NoIndexIteration] or
            any subclass thereof.
        max_in_class_samples: Maximum number of in-class samples to generate per outer
            iteration. Leave as `None` to sample all in-class samples when using finite
            samplers. This must be set to a positive integer when using an infinite
            in-class sampler.
        min_elements_per_label: Minimum number of elements per label
            to sample from the complement set, i.e., out of class elements.
        batch_size: Number of samples to generate in each batch.
    """

    def __init__(
        self,
        in_class: IndexSampler,
        out_of_class: PowersetSampler,
        *,
        max_in_class_samples: int | None = None,
        min_elements_per_label: int = 1,
        batch_size: int = 1,
    ):
        super().__init__(batch_size=batch_size)
        # By default, powerset samplers remove the index from the generated
        # subset but in this case we want all indices.
        # The index for which we compute the value will be removed by
        # the in_class sampler instead.
        if not issubclass(out_of_class._index_iterator_cls, NoIndexIteration):
            raise ValueError(
                "The out-of-class sampler must use NoIndexIteration or any subclass. "
                f"It is currently {out_of_class._index_iterator_cls.__name__}."
            )

        self.in_class = in_class
        self.out_of_class = out_of_class
        self.min_elements_per_label = min_elements_per_label
        self.max_in_class_samples = max_in_class_samples

    def samples_from_data(self, data: Dataset) -> SampleGenerator:
        """Generates batches of class-wise samples from the dataset.
        Args:
            data: The dataset to sample from.
        """
        labels = get_unique_labels(data.data().y)
        n_labels = len(labels)

        if (
            self.max_in_class_samples is None
            and self.in_class.sample_limit(data.indices) is None
        ):
            raise ValueError(
                f"When using an infinite in-class sampler ({str(self.in_class)}), "
                f"ClasswiseSampler.max_in_class_samples must be set to a positive integer "
                f"upon construction."
            )

        out_of_class_batch_generators = {}

        for label in labels:
            without_label = np.where(data.data().y != label)[0]
            out_of_class_batch_generators[label] = self.out_of_class.generate_batches(
                without_label
            )

        for label, ooc_batch in roundrobin(out_of_class_batch_generators):
            if self.interrupted:
                return
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
                for ic_sample in flatten(self.in_class.generate_batches(with_label)):
                    yield ClasswiseSample(
                        idx=ic_sample.idx,
                        label=label,
                        subset=ic_sample.subset,
                        ooc_subset=ooc_sample.subset,
                    )
                    if (
                        self.max_in_class_samples is not None
                        and self.in_class.n_samples >= self.max_in_class_samples
                    ) or self.in_class.interrupted:
                        break

    def batches_from_data(self, data: Dataset) -> BatchGenerator:
        """Batches the samples and yields them."""
        try:
            self._len = self.sample_limit(data.indices)
        except AttributeError:
            pass

        # Create an empty generator if the indices are empty: `return` acts like a
        # `break`, and produces an empty generator.
        if len(data) == 0:
            return

        self._interrupted = False
        self._n_samples = 0
        for batch in chunked(self.samples_from_data(data), self.batch_size):
            self._n_samples += len(batch)
            yield batch
            if self.interrupted:
                break

    def generate_batches(self, indices: IndexSetT) -> BatchGenerator:
        raise AttributeError("Cannot sample from indices directly. Call `from_data()`.")

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        """This is not needed because this sampler is used by calling the `from_data`
        method instead of the `generate_batches` method."""
        raise AttributeError("Cannot sample from indices directly.")

    def log_weight(self, n: int, subset_len: int) -> float:
        """CW-Shapley uses the evaluation strategy from the in-class sampler, so this
        method should never be called."""
        raise AttributeError("The weight should come from the in-class sampler")

    def sample_limit(self, indices: IndexSetT) -> int | None:
        if (
            self.out_of_class.sample_limit(indices)
            or self.in_class.sample_limit(indices) is None
        ):
            return None

        raise AttributeError(
            "When using finite in-class and out-of-class samplers, the sample limit "
            "cannot be computed without the label information."
        )

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: SemivalueCoefficient | None,
    ) -> EvaluationStrategy[IndexSampler, ValueUpdate]:
        return self.in_class.make_strategy(utility, log_coefficient)
