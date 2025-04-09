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

<span id="combinatorial-shapley" class="tm-eqlabel"></span>

$$
\begin{equation}
v_\text{shap}(i) = \frac{1}{n} \sum_{S \subseteq D_{-i}}
\binom{n-1}{ | S | }^{-1} [u(S_{+i}) − u(S)],
  \label{combinatorial-shapley}\tag{1}
\end{equation}
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
    result = valuation.result
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
    scorer = SupervisedScorer("accuracy", test_data, default=0.0)
    utility = ModelUtility(model, scorer)
    sampler = UniformSampler(seed=42)
    stopping = MaxSamples(sampler, 5000)
    valuation = ShapleyValuation(utility, sampler, stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.result
    ```

Note the usage of the object [MaxSamples][pydvl.value.stopping.MaxSamples] as
the stopping condition, which takes the sampler as argument. This is a special
instance of a [StoppingCriterion][pydvl.value.stopping.StoppingCriterion]. More
examples which are not tied to the sampler are
[MaxTime][pydvl.value.stopping.MaxTime] (stops after a certain time),
[MinUpdates][pydvl.value.stopping.MinUpdates] (looks at the number of updates
to the individual values), and
[AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError] (not very
reliable as a stopping criterion), among others.


## A stratified approach  { #stratified-shapley-value }

Let's decompose definition [(1)][combinatorial-shapley-intro] into "layers",
one per subset size $k,$ by writing it in the equivalent form:[^not1]

$$v_\text{shap}(i) = \sum_{k=0}^{n-1} \frac{1}{n} \binom{n-1}{k}^{-1} 
    \sum_{S \subseteq D_{-i}^{k}} \Delta_i(S).$$

Here $D_i^{k}$ is the set of all subsets of size $k$ in the complement  of
$\{i\}.$ Since there are $\binom{n-1}{k}$ such sets, the above is an average
over all $n$ set sizes $k$ of the average marginal contributions of the point
$i$ to all sets of size $k.$

We can now devise a sampling scheme over the powerset of $N_{-i}$ that yields
this expression:

1. Sample $k$ uniformly from $\{0, ..., n-1\}.$
2. Sample $S$ uniformly from the powerset of $N_{-i}^k.$

Call this distribution $\mathcal{L}_k.$ Then

$$
\begin{eqnarray*}
    \mathbb{E}_{S \sim \mathcal{L}} [\Delta_i (S)] 
            & = & \sum_{k = 0}^{n - 1} \sum_{S \subseteq N_{- i}^k} 
                  \Delta_i (S) p (S|k) p (k) \\
            & = & \sum_{k = 0}^{n - 1} \sum_{S \subseteq N_{- i}^k} \Delta_i (S)
                  \binom{n - 1}{k}^{- 1} \frac{1}{n} \\
            & = & v_{\text{sh}}(i).
\end{eqnarray*}
$$

The choice $p(k) = 1/n$ is implemented in 
[StratifiedShapleyValuation][pydvl.valuation.methods.shapley.StratifiedShapleyValuation]
but can be changed to any other distribution over $k.$ [@wu_variance_2023]
introduced [VRDS sampling][pydvl.valuation.samplers.stratified.VRDSSampler] as
a way to reduce the variance of the estimator.

??? Example "Stratified Shapley"
    The specific instance of stratified sampling described above can be directly
    used by instantiating a
    [StratifiedShapleyValuation][pydvl.valuation.methods.shapley.StratifiedShapleyValuation]
    object. For more general use cases, use
    [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] with a
    custom sampler, for instance
    [VRDSSampler][pydvl.valuation.samplers.stratified.VRDSSampler].
    Note the use of the [History][pydvl.value.stopping.History] object, a stopping
    which does not stop, but records the trace of value updates in a rolling
    memory. The data can then be used to check for convergence, debugging,
    plotting, etc.

    ```python
    from pydvl.valuation import StratifiedShapleyValuation, MinUpdates, History
    training_data, test_data = Dataset.from_arrays(...)
    model = ...
    scorer = SupervisedScorer(model, test_data, default=..., range=...)
    utility = ModelUtility(model, scorer)
    valuation = StratifiedShapleyValuation(
        utility=utility,
        is_done=MinUpdates(min_updates) | History(n_steps=min_updates),
        batch_size=batch_size,
        seed=seed,
        skip_converged=True,
        progress=True,
    )
    with parallel_config(n_jobs=-4):
        valuation.fit(training_data)
    results = valuation.result
    ```

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
    Use of this object follows the same pattern as the previous examples, except
    that separate instantiation of the sampler is not necessary anymore. This
    has the drawback that we cannot use
    [MaxSamples][pydvl.value.stopping.MaxSamples] as stopping criterion anymore
    since it requires the sampler. To work around this, use
    [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
    directly.

    ```python
    from pydvl.valuation import (
        MinUpdates
        ModelUtility,
        PermutationSampler,
        SupervisedScorer,
        RelativeTruncation,
        TMCShapleyValuation,
    )

    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test_data, default=0)
    utility = ModelUtility(model, scorer, ...)
    truncation = RelativeTruncation(rtol=0.05)
    stopping = MinUpdates(5000)
    valuation = TMCShapleyValuation(utility, truncation, stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.result
    ```

## Other approximation methods

As already mentioned, with the architecture of
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] it is
possible to try different importance-sampling schemes by swapping the sampler.
Besides TMCS we also have [Owen sampling][owen-shapley-intro]
[@okhrati_multilinear_2021], and [Beta Shapley][beta-shapley-intro]
[@kwon_beta_2022] when $\alpha = \beta = 1.$

A different approach is via a SAT problem, as done in [Group Testing
Shapley][group-testing-shapley-intro] [@jia_efficient_2019].

Yet another, which is applicable to any utility-based valuation method, is [Data
Utility Learning][data-utility-learning-intro] [@wang_improving_2022]. This
method learns a model of the utility function during a warmup phase, and then
uses it to speed up marginal utility computations.


## Model-specific methods

Shapley values can have a closed form expression or a simpler approximation
scheme when the model class is restricted. The prime example is
[kNN-Shapley][knn-shapley-intro] [@jia_efficient_2019a], which is exact for the
kNN model, and is $O(n_\text{test}\  n \log n).$

[^not1]: The quantity $u(S_{+i}) − u(S)$ is called the
  [marginal utility][glossary-marginal-utility] of the sample $x_i$ (with
  respect to $S$), and we will often denote it by $\Delta_i(S, u),$ or, when no
  confusion is possible, simply $\Delta_i(S).$
