r"""
Samplers iterate over subsets of indices.

The classes in this module are used to iterate over indices, and subsets of their
complement in the whole set, as required for the computation of marginal utilities
for semi-values and other marginal-utility based methods.

Subclasses of [IndexSampler][pydvl.valuation.samplers.IndexSampler] are iterators
over **batches** of [Samples][pydvl.valuation.samplers.Sample] of the form $(i, S)$, where
$i$ is an index of interest, and $S \subset I \setminus \{i\}$ is a subset of the
complement of $i$.

The samplers are used by all game-theoretic valuation methods, as well as for LOO and
any other marginal-contribution-based method which iterates over subsets of the training
data.

## Sampler evaluation

Because different samplers require different strategies for evaluating the utility
of the subsets, the samplers are used in conjunction with an
[EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]. The
basic usage pattern inside a valuation method is the following:

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

See more on the [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
class.

## Creating custom samplers

To create a custom sampler, subclass either
[PowersetSampler][pydvl.valuation.samplers.PowersetSampler]
or [PermutationSampler][pydvl.valuation.samplers.PermutationSampler], or
implement the [IndexSampler][pydvl.valuation.samplers.IndexSampler] interface directly.

There are two main methods to implement (and others that can be overridden):

* [generate()][pydvl.valuation.samplers.IndexSampler.generate], which yields samples of the
  form $(i, S)$. These will be batched together by `__iter__`. For `PermutationSampler`,
  the batch size is always the number of indices since permutations must always be
  processed in full.
* [weight()][pydvl.valuation.samplers.IndexSampler.weight] to provide a factor by which to
  multiply Monte Carlo samples in stochastic methods, so that the mean converges to the
  desired expression.

Additionally, if the sampler requires a dedicated evaluation strategy different from
the marginal evaluations for `PowersetSampler` or the successive evaluations for
`PermutationSampler`, you need to subclass
[EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] and set the
`strategy_cls` attribute of the sampler to this class.

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
    UniformStratifiedSampler,
    TruncatedUniformStratifiedSampler,
    StratifiedSampler,
    OwenSampler,
    AntitheticOwenSampler,
]
