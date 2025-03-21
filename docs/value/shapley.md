---
title: Shapley value
alias:
  name: shapley-valuation-intro
  title: Shapley value
---

# Shapley value  { #shapley-valuation-intro }

The Shapley method is an approach to compute data values originating in
cooperative game theory. Shapley values are a common way of assigning payoffs to
each participant in a cooperative game (i.e. one in which players can form
coalitions) in a way that ensures that certain axioms are fulfilled.

pyDVL implements several methods for the computation and approximation of
Shapley values. Empirically, one of the most useful methods is the so-called 
[Truncated Monte Carlo Shapley][tmcs-intro] [@ghorbani_data_2019], but several
approximations exist with different convergence rates and computational costs.


## Combinatorial Shapley  { #combinatorial-shapley-intro }

The first algorithm is just a verbatim implementation of the definition below.
As such it returns as exact a value as the utility function allows (see what
this means in [Problems of Data Values][problems-of-data-values]).

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
as a weighted sum of its marginal utility wrt. every possible coalition of
training samples within the training set:

$$
v(i) = \frac{1}{n} \sum_{S \subseteq D_{-i}}
\binom{n-1}{ | S | }^{-1} [u(S_{+i}) − u(S)],
$$

where $D_{-i}$ denotes the set of samples in $D$ excluding $x_i,$ and $S_{+i}$
denotes the set $S$ with $x_i$ added.[^not1]

??? example "Computing exact Shapley values"
    ```python
    from joblib import parallel_config
    from pydvl.valuation import (
        Dataset, ModelUtility, SupervisedScorer, ShapleyValuation
    )

    train, test = SomeVerySmallDatasets()
    model = ...
    scorer = SupervisedScorer(model, test, default=..)
    utility = ModelUtility(model, scorer)
    sampler = DeterministicUniformSampler()
    valuation = ShapleyValuation(utility, sampler, NoStopping(sampler))

    with parallel_config(n_jobs=-1):
        valuation.fit(train)
    result = valuation.values()
    ```

We can convert the return value to a
[pandas.DataFrame][] with the `to_dataframe` method. Please refer to the
[introduction to data valuation][data-valuation-intro] and to the documentation
in [ValuationResult][pydvl.valuation.result.ValuationResult] for more
information.

## Monte Carlo Combinatorial Shapley  { #monte-carlo-combinatorial-shapley-intro }

Because the number of subsets $S \subseteq D_{-i}$ is $2^{ | D | - 1 },$ one
must typically resort to approximations. The simplest one is done via Monte
Carlo sampling of the powerset $\mathcal{P}(D).$ In pyDVL this simple technique
is called "Monte Carlo Combinatorial". The method has very poor converge rate
and others are preferred, but if desired, usage follows the same pattern:

??? example "Monte Carlo Combinatorial Shapley values"
    ```python
    from pydvl.valuation import (
        ShapleyValuation,
        ModelUtility,
        SupervisedScorer,
        PermutationSampler,
        MaxSamples
    )

    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test_data, default=0)
    utility = ModelUtility(model, scorer, ...)
    sampler = UniformSampler(seed=42)
    stopping = MaxSamples(5000)
    valuation = ShapleyValuation(utility, sampler, stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.values()
    ```

The DataFrames returned by most Monte Carlo methods will contain approximate
standard errors as an additional column, in this case named `cmc_stderr`.

Note the usage of the object [MaxUpdates][pydvl.value.stopping.MaxUpdates] as the
stop condition. This is an instance of a
[StoppingCriterion][pydvl.value.stopping.StoppingCriterion]. Other examples are
[MaxTime][pydvl.value.stopping.MaxTime] and
[AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError].


## Permutation Shapley  { #permutation-shapley-intro }

An equivalent way of computing Shapley values (`ApproShapley`) appeared in
[@castro_polynomial_2009] and is the basis for the method most often used in
practice. It uses permutations over indices instead of subsets:

$$
v_u(i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
[u(S_{i}^{\sigma} \cup \{i\}) − u(S_{i}^{\sigma})],
$$

where $S_{i}^{\sigma}$ denotes the set of indices in permutation sigma before the
position where $i$ appears. To approximate this sum (which has $\mathcal{O}(n!)$
terms!) one uses Monte Carlo sampling of permutations, something which has
surprisingly low sample complexity. One notable difference wrt. the
combinatorial approach above is that the approximations always fulfill the
efficiency axiom of Shapley, namely $\sum_{i=1}^n \hat{v}_i = u(D)$ (see
[@castro_polynomial_2009], Proposition 3.2).

??? info "A note about implementation"
    The definition above uses all permutations to update one datapoint $i$.
    However, in practice, instead of searching for the position of a fixed index
    in every permutation, one can use a single permutation to update all
    datapoints, by iterating through it and updating the value for the index at
    the current position. This has the added benefit of allowing to use the
    utility for the previous index to compute the marginal utility for the
    current one, thus halving the number of utility calls. This strategy is
    implemented in
    [PermutationEvaluationStrategy][pydvl.valuation.samplers.permutation.PermutationEvaluationStrategy],
    and is automatically selected when using any of the permutation samplers.


## Truncated Monte Carlo Shapley { #tmcs-intro }

By adding two types of early stopping, the result is the so-called **Truncated
Monte Carlo Shapley** [@ghorbani_data_2019], which is efficient enough to be
useful in applications. 

The first is simply a convergence criterion, of which
there are [several to choose from][pydvl.value.stopping]. The second is a
criterion to truncate the iteration over single permutations.
[RelativeTruncation][pydvl.value.shapley.truncated.RelativeTruncation] chooses
to stop iterating over samples in a permutation when the marginal utility
becomes too small. The method is available through the class
[TMCShapleyValuation][pydvl.valuation.methods.shapley.TMCShapleyValuation].

However, being a heuristic to permutation sampling, it can be "manually"
implemented by choosing a
[RelativeTruncation][pydvl.valuation.samplers.truncation.RelativeTruncation]
for a
[PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
when configuring
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
(note however that this introduces some correcting factors, see 
[[semi-values-sampling]]).

You can see this method in action in
[this example](../../examples/shapley_basic_spotify/) using the Spotify dataset.


??? example "Truncated Monte Carlo Shapley values"
    ```python
    from pydvl.valuation import (
        TMCShapleyValuation,
        ModelUtility,
        SupervisedScorer,
        PermutationSampler,
        RelativeTruncation,
        MaxSamples
    )

    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test_data, default=0)
    utility = ModelUtility(model, scorer, ...)
    truncation = RelativeTruncation(rtol=0.05)
    stopping = MaxSamples(5000)
    valuation = TMCShapleyValuation(utility, truncation, stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.values()
    ```

## Other approximation methods

As already mentioned, with the architecture of
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] it is
possible to try different importance-sampling schemes by swapping the sampler.
Besides TMCS we also have [Owen sampling][owen-shapley-intro]
[@okhrati_multilinear_2021], and [Beta
Shapley][beta-shapley-intro] [@kwon_beta_2022] when $\alpha = \beta = 1.$

A different approach is via a SAT problem, as done in [Group Testing
Shapley][group-testing-shapley-intro] [@jia_efficient_2019].

Yet another, which is applicable to any utility-based valuation method, is
[Data Utility Learning][data-utility-learning-intro]
[@wang_improving_2022]. This method learns a model of the utility function
during a warmup phase, and then uses it to speed up marginal utility
computations.


## Model-specific methods

Shapley values can have a closed form expression or a simpler approximation
scheme when the model class is restricted. The prime example is
[kNN-Shapley][knn-shapley-intro] [@jia_efficient_2019a], which is exact for the
kNN model, and is $O(n_test n \log n).$

[^not1]: The quantity $u(S_{+i}) − u(S)$ is called the
  [marginal utility][glossary-marginal-utility] of the sample $x_i$ (with
  respect to $S$), and we will often denote it by $\delta_i(S, u),$ or, when no
  confusion is possible, simply $\delta_i(S).$
