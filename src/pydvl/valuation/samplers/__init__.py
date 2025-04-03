r"""
Samplers iterate over subsets of indices.

The classes in this module are used to iterate over sets of indices, as required for the
computation of marginal utilities for [semi-values][semi-values-intro] and other
[marginal-utility][glossary-marginal-utility] based methods, in particular all
[game-theoretic methods][glossary-game-theoretic-methods]. Because of the intertwining
of these algorithms with the sampler employed, there are several strategies to choose
when deploying, or extending each.

## A user's guide

* Construct a sampler by instantiating one of the classes in this module. Refer to
  the documentation of each class for details.
* Pass the constructed sampler to the method. Not all combinations of sampler and
  valuation method are meaningful.
* When using finite samplers, use the [NoStopping][pydvl.valuation.stopping.NoStopping]
  criterion and pass it the sampler to keep track of progress.


## A high-level overview

Subclasses of [IndexSampler][pydvl.valuation.samplers.base.IndexSampler] are iterators
over **batches** of [Samples][pydvl.valuation.types.Sample]. Each sample is typically,
but not necessarily, of the form $(i, S)$, where $i$ is an index of interest, and $S
\subseteq N \setminus \{i\}$ is a subset of the complement of $i$ over the index set
$N.$

Samplers reside in the main process. Their samples are sent to the workers by the
`fit()` method of the valuation class, to be processed by a so-called
[EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]'s `process()`
method. These strategies return [ValueUpdate][pydvl.valuation.types.ValueUpdate]
objects, which are then aggregated into the final result by the main process.


## Sampler weights

Because the samplers are used in a Monte Carlo setting, they can be weighted to perform
importance sampling. To this end, classes inheriting from
[IndexSampler][pydvl.valuation.samplers.base.IndexSampler] implement the
[log_weight()][pydvl.valuation.samplers.base.IndexSampler.log_weight] method, which
returns the (logarithm of) the probability of sampling a given subset. This is used to
correct the mean of the Monte Carlo samples, so that it converges to the desired
expression. For an explanation of the interactions between sampler weights, semi-value
coefficients and importance sampling, see [Sampling strategies for
semi-values][semi-values-sampling].

## Sampler evaluation

Different samplers require different strategies for processing samples, i.e. for
evaluating the utility of the subsets. For instance, [permutation
samplers][pydvl.valuation.samplers.permutation] generate full permutations of the index
set, and rely on a special evaluation loop that allows semi-value calculations to reuse
computations, by iterating through each permutation sequentially.

This behaviour is encoded in the
[EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] class, which the
evaluation method retrieves through
[make_strategy()][pydvl.valuation.samplers.base.IndexSampler.make_strategy].

??? info "Usage pattern in valuation method"
    The basic pattern is the following (see below for info on the `updater`):
    ```python
        def fit(self, data: Dataset):

            ...

            strategy = self.sampler.make_strategy(self.utility, self.log_coefficient)
            processor = delayed(strategy.process)
            updater = self.sampler.result_updater(self.result)

            delayed_batches = Parallel()(
                processor(batch=list(batch), is_interrupted=flag) for batch in self.sampler
            )
            for batch in delayed_batches:
                for evaluation in batch:
                    self.result = updater(evaluation)
                ...
    ```

## Updating the result

Yet another behaviour that depends on the sampling scheme is the way that results are
updated. For instance, the [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler]
requires tracking updates to two sequences of samples which are then merged in a
specific way. This strategy is declared by the sampler through the factory method
[result_updater()][pydvl.valuation.samplers.base.IndexSampler.result_updater],
which returns a callable that updates the result with a single evaluation.


## Creating custom samplers

To create a custom sampler, subclass either
[PowersetSampler][pydvl.valuation.samplers.PowersetSampler]
or [PermutationSamplerBase][pydvl.valuation.samplers.permutation.PermutationSamplerBase],
or implement the [IndexSampler][pydvl.valuation.samplers.IndexSampler] interface
directly.

There are three main methods to implement (and others that can be overridden):

* [generate()][pydvl.valuation.samplers.base.IndexSampler.generate], which yields
  samples of the form $(i, S)$. These will be batched together by `__iter__` for
  parallel processing. Note that, if the index set has size $N$, for
  [PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler], a
  batch size of $B$ implies $O(B*N)$ evaluations of the utility in one process, since
  single permutations are always processed in one go.
* [log_weight()][pydvl.valuation.samplers.base.IndexSampler.log_weight] to provide a
  factor by which to multiply Monte Carlo samples in stochastic methods, so that the
  mean converges to the desired expression. This will be the logarithm of the
  probability of sampling a given subset. For an explanation of the interactions between
  sampler weights, semi-value coefficients and importance sampling, see
  [Sampling strategies for semi-values][semi-values-sampling].

    ??? tip "Disabling importance sampling"
        If you want to disable importance sampling, you can override the property
        [log_coefficient()][pydvl.valuation.methods.semivalue.SemivalueValuation.log_coefficient]
        and return `None`. This will make the evaluation strategy ignore the sampler
        weights and the Monte Carlo sums converge to the expectation of the marginal
        utilities wrt. the sampling distribution, with no change.
* [sample_limit()][pydvl.valuation.samplers.base.IndexSampler.sample_limit] to return
  the maximum number of samples that can be generated from a set of indices. Infinte
  samplers should return `None`.
* [make_strategy()][pydvl.valuation.samplers.base.IndexSampler.make_strategy] to create
  an evaluation strategy that processes the samples. This is typically a subclass of
  [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] that computes
  utilities and weights them with coefficients and sampler weights.
  One can also use any of the predefined strategies, like the successive marginal
  evaluations of
  [PowersetEvaluationStrategy][pydvl.valuation.samplers.powerset.PowersetEvaluationStrategy]
  or the successive evaluations of
  [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy].

Finally, if the sampler requires a dedicated result updater, you must override
[result_updater()][pydvl.valuation.samplers.base.IndexSampler.result_updater] to return
a callable that updates a [ValuationResult][pydvl.valuation.result.ValuationResult] with
one evaluation [ValueUpdate][pydvl.valuation.types.ValueUpdate]. This is used e.g. for
the [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] which uses two running means
for positive and negative updates.

!!! tip "Changed in version 0.10.0"
    All the samplers in this module have been changed to work with the new
    evaluation strategies.

## References

[^1]: <a name="mitchell_sampling_2022"></a>Mitchell, Rory, Joshua Cooper, Eibe
      Frank, and Geoffrey Holmes. [Sampling Permutations for Shapley Value
      Estimation](https://jmlr.org/papers/v23/21-0439.html). Journal of Machine
      Learning Research 23, no. 43 (2022): 1–46.
[^2]: <a name="watson_accelerated_2023"></a>Watson, Lauren, Zeno Kujawa, Rayna Andreeva,
      Hao-Tsung Yang, Tariq Elahi, and Rik Sarkar. [Accelerated Shapley Value
      Approximation for Data Evaluation](https://doi.org/10.48550/arXiv.2311.05346).
      arXiv, 9 November 2023.
"""

from typing import Union

from .base import *
from .classwise import *
from .msr import *
from .owen import *
from .permutation import *
from .powerset import *
from .stratified import *
from .truncation import *

# TODO Replace by Intersection[StochasticSamplerMixin, PowersetSampler[T]]
# See https://github.com/python/typing/issues/213
StochasticSampler = Union[
    UniformSampler,
    PermutationSampler,
    AntitheticSampler,
    StratifiedSampler,
    OwenSampler,
    AntitheticOwenSampler,
]
