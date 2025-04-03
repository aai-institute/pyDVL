---
title: Leave-One-Out values
alias: 
  name: loo-valuation-intro
---

# Leave-One-Out values  { #loo-valuation-intro }

LOO is the simplest approach to valuation. Let $D$ be the training set, and
$D_{-i}$ be the training set without the sample $x_i$. Assume some utility
function $u(S)$ that measures the performance of a model trained on
$S \subseteq D$.

LOO assigns to each sample its *marginal utility* as value:

$$v_\text{loo}(i) = u(D) - u(D_{-i}),$$

and as such is the simplest example of marginal contribution-based valuation
method. In pyDVL it is available as
[LOOValuation][pydvl.valuation.methods.loo.LOOValuation].

For the purposes of data valuation, this is rarely useful beyond serving as a
baseline for benchmarking (although it can perform astonishingly well on
occasion).

One particular weakness is that it does not necessarily correlate with an
intrinsic value of a sample: since it is a marginal utility, it is affected by
_diminishing returns_. Often, the training set is large enough for a single
sample not to have any significant effect on training performance, despite any
qualities it may possess. Whether this is indicative of low value or not depends
on one's goals and definitions, but other methods are typically preferable.

??? Example "Leave-One-Out values"
    ```python
    from joblib import parallel_config
    from pydvl.valuation import Dataset, LOOValuation, ModelUtility
    
    train, test = Dataset.from_arrays(...)
    model = ...
    utility = ModelUtility(model)
    val_loo = LOOValuation(utility, progress=True)
    with parallel_config(n_jobs=12):
        val_loo.fit(train)
    result = val_loo.values(sort=True)
    ```

Strictly speaking, LOO can be seen as a [semivalue][semi-values-intro] where
all the coefficients are zero except for $k=|D|-1.$

!!! tip "Connection to the influence function"
    With a slight change of perspective, the _influence function_ can be seen as
    a first order approximation to the Leave-One-Out values. See [Approximating
    the influence of a point][influence-of-a-point].
