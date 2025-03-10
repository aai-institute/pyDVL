---
title: Beta Shapley
alias:
  name: beta-shapley-intro
  text: Introduction to Beta Shapley
---

# Beta Shapley  { #beta-shapley-intro }

In ML applications, where the utility is the performance when trained on a set
$S \subset D$, diminishing returns are often observed when computing the
marginal utility of adding a new data point.[^diminishing-returns]

Beta Shapley is a weighting scheme that uses the Beta function to place more
weight on subsets deemed to be more informative. The weights are defined as:

$$
w(k) := \frac{B(k+\beta, n-k+1+\alpha)}{B(\alpha, \beta)},
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function),
and $\alpha$ and $\beta$ are parameters that control the weighting of the
subsets. Setting both to 1 recovers Shapley values, and setting $\alpha = 1$,
and $\beta = 16$ is reported in [@kwon_beta_2022] to be a good choice for some
applications. Beta Shapley values are available in pyDVL through
[BetaShapleyValuation][pydvl.valuation.methods.beta_shapley.BetaShapleyValuation]:

??? Example "Beta Shapley values"
    ```python
    from joblib import parallel_config
    from pydvl.valuation import *
    
    model = ...
    train, test = Dataset.from_arrays(...)
    scorer = SupervisedScorer(model, test, default=0.0)
    utility = ModelUtility(model, scorer)
    sampler = PermutationSampler()
    stopping = RankCorrelation(rtol=1e-5, burn_in=100) | MaxUpdates(2000)
    valuation = BetaShapleyValuation(
        utility, sampler, stopping, alpha=1, beta=16
    )
    with parallel_config(n_jobs=16):
        valuation.fit(train)
    ```

See, however [[data-banzhaf-intro|Banzhaf indices]], for an alternative
choice of weights which is reported to work better in cases of high variance in
the utility function.

[^diminishing-returns]: This observation is made somewhat formal for some 
    model classes in [@watson_accelerated_2023], motivating a complete
    truncation of the sampling space, see [$\delta$-Shapley][delta-shapley-intro].
