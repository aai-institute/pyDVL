"""
Base classes for samplers and evaluation strategies.

See [pydvl.valuation.samplers][pydvl.valuation.samplers] for details.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
from more_itertools import chunked

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import (
    BatchGenerator,
    IndexSetT,
    NullaryPredicate,
    SampleBatch,
    SampleGenerator,
    ValueUpdateT,
)
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["EvaluationStrategy", "IndexSampler", "ResultUpdater"]


# Sequence.register(np.ndarray)  # <- Doesn't seem to work

logger = logging.getLogger(__name__)


class ResultUpdater(Protocol[ValueUpdateT]):
    """Protocol for result updaters."""

    def __call__(self, update: ValueUpdateT) -> ValuationResult: ...


class IndexSampler(ABC, Generic[ValueUpdateT]):
    r"""Samplers are custom iterables over batches of subsets of indices.

    Calling `from_indices(indexset)` on a sampler returns a generator over **batches**
    of `Samples`. A [Sample][pydvl.valuation.samplers.Sample] is a tuple of the form
    $(i, S)$, where $i$ is an index of interest, and $S \subset I \setminus \{i\}$ is a
    subset of the complement of $i$ in $I$.

    !!! Note
        Samplers are **not** iterators themselves, so that each call to
        `from_indices(data)` e.g. in a new for loop creates a new iterator.

    Derived samplers must implement
    [weight()][pydvl.valuation.samplers.IndexSampler.weight] and
    [_generate()][pydvl.valuation.samplers.IndexSampler._generate]. See the module's
    documentation for more on these.

    ## Interrupting samplers

    Calling [interrupt()][pydvl.valuation.samplers.IndexSampler.interrupt] on a sampler
    will stop the batched generator after the current batch has been yielded.

    Args:
        batch_size: The number of samples to generate per batch. Batches are
            processed by
            [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
            so that individual valuations in batch are guaranteed to be received in the
            right sequence.

    ??? Example
        ``` pycon
        >>>from pydvl.valuation.samplers import DeterministicUniformSampler
        >>>import numpy as np
        >>>sampler = DeterministicUniformSampler()
        >>>for idx, s in sampler.generate_batches(np.arange(2)):
        >>>    print(s, end="")
        [][2,][][1,]
        ```
    """

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: The number of samples to generate per batch. Batches are
                processed by the
                [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
        """
        self._batch_size = batch_size
        self._n_samples = 0
        self._interrupted = False

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int) -> None:
        raise AttributeError("Cannot reset a sampler's number of samples")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if value < 1:
            raise ValueError("batch_size must be at least 1")
        self._batch_size = value

    def interrupt(self):
        self._interrupted = True

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def from_data(self, data: Dataset) -> BatchGenerator:
        return self.generate_batches(data.indices)

    def generate_batches(self, indices: IndexSetT) -> BatchGenerator:
        """Batches the samples and yields them."""

        # create an empty generator if the indices are empty. `generate_batches` is
        # a generator function because it has a yield statement later in its body.
        # Inside generator function, `return` acts like a `break`, which produces an
        # empty generator function. See: https://stackoverflow.com/a/13243870
        if len(indices) == 0:
            return

        self._interrupted = False
        self._n_samples = 0
        for batch in chunked(self._generate(indices), self.batch_size):
            yield batch
            self._n_samples += len(batch)
            if self._interrupted:
                break

    def sample_limit(self, indices: IndexSetT) -> int | None:
        """Number of samples that can be generated from the indices.

        Returns None if the number of samples is infinite, which is the case for most
        stochastic samplers.
        """
        if len(indices) == 0:
            out = 0
        else:
            out = None
        return out

    @abstractmethod
    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        """Generates single samples.

        `IndexSampler.generate_batches()` will batch these samples according to the
        batch size set upon construction.

        Args:
            indices:

        Yields:
            A tuple (idx, subset) for each sample.
        """
        ...

    @staticmethod
    @abstractmethod
    def weight(n: int, subset_len: int) -> float:
        r"""Factor by which to multiply Monte Carlo samples, so that the
        mean converges to the desired expression.

        By the Law of Large Numbers, the sample mean of $\delta_i(S_j)$
        converges to the expectation under the distribution from which $S_j$ is
        sampled.

        $$
        \frac{1}{m}  \sum_{j = 1}^m \delta_i (S_j) c (S_j) \longrightarrow
           \underset{S \sim \mathcal{D}_{- i}}{\mathbb{E}} [\delta_i (S) c ( S)]
        $$

        We add a factor $c(S_j)$ in order to have this expectation coincide with
        the desired expression.

        Args:
            n: The total number of indices in the training data.
            subset_len: The size of the subset $S_j$ for which the marginal is being
                computed
        """
        ...

    @abstractmethod
    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ) -> EvaluationStrategy:
        """Returns the strategy for this sampler."""
        ...  # return SomeEvaluationStrategy(self)

    def result_updater(self, result: ValuationResult) -> ResultUpdater[ValueUpdateT]:
        """Returns a function that updates a valuation result with a value update.

        This is typically just the method provided by the
        [ValuationResult][pydvl.valuation.result.ValuationResult] class, but some
        algorithms might require a different update strategy.

        Args:
            result: The result to update
        Returns:
            A callable object that updates the result with a value update
        """

        def updater(update: ValueUpdateT) -> ValuationResult:
            assert update.idx is not None
            result.update(update.idx, update.update)
            return result

        return updater


SamplerT = TypeVar("SamplerT", bound=IndexSampler)


class EvaluationStrategy(ABC, Generic[SamplerT, ValueUpdateT]):
    """An evaluation strategy for samplers.

    Implements the processing strategy for batches returned by an
    [IndexSampler][pydvl.valuation.samplers.IndexSampler].

    Different sampling schemes require different strategies for the evaluation of the
    utilities. For instance permutations generated by
    [PermutationSampler][pydvl.valuation.samplers.PermutationSampler] must be evaluated
    in sequence to save computation, see
    [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy].

    This class defines the common interface.

    ??? Example "Usage pattern in valuation methods"
        ```python
            def fit(self, data: Dataset):
                self.utility.training_data = data
                strategy = self.sampler.strategy(self.utility, self.coefficient)
                delayed_batches = Parallel()(
                    delayed(strategy.process)(batch=list(batch), is_interrupted=flag)
                    for batch in self.sampler
                )
                for batch in delayed_batches:
                    for evaluation in batch:
                        self.result.update(evaluation.idx, evaluation.update)
                    if self.is_done(self.result):
                        flag.set()
                        break
        ```
    FIXME the coefficient does not belong here, but in the methods. Either we
      return more information from process() so that it can be used in the methods
      or we allow for some manipulation of the strategy after it has been created.
      The latter is rigid but a quick fix, which I need right now.

    Args:
        sampler: Required to setup some strategies. Be careful not to store it in the
            object when subclassing!
        utility: Required to setup some strategies and to process the samples. Since
            this contains the training data, it is expensive to pickle and send to
            workers.
        coefficient: An additional coefficient to multiply marginals with. This
            depends on the valuation method, hence the delayed setup.
    """

    def __init__(
        self,
        sampler: SamplerT,
        utility: UtilityBase,
        coefficient: Callable[[int, int, float], float] | None = None,
    ):
        self.utility = utility
        self.n_indices = (
            len(utility.training_data) if utility.training_data is not None else 0
        )
        self.coefficient: Callable[[int, int], float] = lambda n, k: 1.0

        if coefficient is not None:

            def coefficient_fun(n: int, subset_len: int) -> float:
                return coefficient(n, subset_len, sampler.weight(n, subset_len))

            self.coefficient = coefficient_fun
        else:
            self.coefficient = sampler.weight

    @abstractmethod
    def process(
        self, batch: SampleBatch, is_interrupted: NullaryPredicate
    ) -> list[ValueUpdateT]:
        """Processes batches of samples using the evaluator, with the strategy
        required for the sampler.

        !!! Warning
            This method is intended to be used by the evaluator to process the samples
            in one batch, which means it might be sent to another process. Be careful
            with the objects you use here, as they will be pickled and sent over the
            wire.

        Args:
            batch: A batch of samples to process.
            is_interrupted: A predicate that returns True if the processing should be
                interrupted.

        Yields:
            Updates to values as tuples (idx, update)
        """
        ...
