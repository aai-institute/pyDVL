---
title: Semi-values
---

# Semi-values

SV is a particular case of a more general concept called semi-value,
which is a generalization to different weighting schemes. A **semi-value** is
any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D_{-i}^{(k)}} [u(S_{+i}) - u(S)],
$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1,$$

the set $D_{-i}^{(k)}$ contains all subsets of $D$ of size $k$ that do not
include sample $x_i$, $S_{+i}$ is the set $S$ with $x_i$ added, and $u$ is the
utility function.

Two instances of this are **Banzhaf indices** [@wang_data_2022],
and **Beta Shapley** [@kwon_beta_2022], with better numerical and
rank stability in certain situations.

!!! Note
    Shapley values are a particular case of semi-values and can therefore also
    be computed with the methods described here. However, as of version 0.6.0,
    we recommend using
    [compute_shapley_values][pydvl.value.shapley.compute_shapley_values]
    instead, in particular because it implements truncation policies for TMCS.


## Beta Shapley

For some machine learning applications, where the utility is typically the
performance when trained on a set $S \subset D$, diminishing returns are often
observed when computing the marginal utility of adding a new data point.

Beta Shapley is a weighting scheme that uses the Beta function to place more
weight on subsets deemed to be more informative. The weights are defined as:

$$
w(k) := \frac{B(k+\beta, n-k+1+\alpha)}{B(\alpha, \beta)},
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function),
and $\alpha$ and $\beta$ are parameters that control the weighting of the
subsets. Setting both to 1 recovers Shapley values, and setting $\alpha = 1$, and
$\beta = 16$ is reported in [@kwon_beta_2022] to be a good choice for
some applications. See however the [Banzhaf indices][banzhaf-indices] section 
for an alternative choice of weights which is reported to work better.

```python
from pydvl.value import *

utility = Utility(model, data)
values = compute_beta_shapley_semivalues(
    u=utility, done=AbsoluteStandardError(threshold=1e-4), alpha=1, beta=16
)
```

## Banzhaf indices

As noted in the section [Problems of Data Values][problems-of-data-values],
the Shapley value can be very sensitive to variance in the utility function.
For machine learning applications, where the utility is typically the performance
when trained on a set $S \subset D$, this variance is often largest
for smaller subsets $S$. It is therefore reasonable to try reducing
the relative contribution of these subsets with adequate weights.

One such choice of weights is the Banzhaf index, which is defined as the
constant:

$$w(k) := 2^{n-1},$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

The authors of [@wang_data_2022] show that Banzhaf indices are more
robust to variance in the utility function than Shapley and Beta Shapley values.

```python
from pydvl.value import *

utility = Utility(model, data)
values = compute_banzhaf_semivalues(
    u=utility, done=AbsoluteStandardError(threshold=1e-4), alpha=1, beta=16
)
```

## General semi-values

As explained above, both Beta Shapley and Banzhaf indices are special cases of
semi-values. In pyDVL we provide a general method for computing these with any
combination of the three ingredients that define a semi-value:

- A utility function $u$.
- A sampling method
- A weighting scheme $w$.

The utility function is the same as for Shapley values, and the sampling method
can be any of the types defined in [the samplers module][pydvl.value.sampler].
For instance, the following snippet is equivalent to the above:

```python
from pydvl.value import *

data = Dataset(...)
utility = Utility(model, data)
values = semivalues(
    sampler=PermutationSampler(data.indices),
    u=utility,
    coefficient=beta_coefficient(alpha=1, beta=16),
    done=AbsoluteStandardError(threshold=1e-4),
  )
```

!!! warning "Careful with permutation sampling"
    This generic implementation of semi-values allowing for any combination of
    sampling and weighting schemes is very flexible and, in principle, it
    recovers the original Shapley value, so that 
    [compute_shapley_values][pydvl.value.shapley.common.compute_shapley_values]
    is no longer necessary. However, it loses the optimization in permutation
    sampling that reuses the utility computation from the last iteration when
    iterating over a permutation. This doubles the computation requirements (and
    slightly increases variance) when using permutation sampling, unless [the
    cache](getting-started/installation.md#setting-up-the-cache) is enabled.
    In addition,
    [truncation policies][pydvl.value.shapley.truncated.TruncationPolicy] are
    not supported for in this generic implementation (as of v0.7.0). For these
    reasons it is preferable to use
    [compute_shapley_values][pydvl.value.shapley.common.compute_shapley_values]
    whenever not computing other semi-values.
