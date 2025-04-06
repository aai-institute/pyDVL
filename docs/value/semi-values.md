---
title: Semi-values
alias:
  name: semi-values-intro
---

# Semi-values  { #semi-values-intro }

The well-known [Shapley Value][shapley-valuation-intro] is a particular case of a
more general concept called semi-value, which is a generalization to different
weighting schemes. A **semi-value** is any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subseteq D_{-i}^{k}} [u(S_{+i}) - u(S)],
$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n \binom{n-1}{k} w(k) = 1,$$

and $D_{-i}^{k}$ is the set of all sets $S$ of size $k$ that do not include
sample $x_i$, $S_{+i}$ is the set $S$ with $x_i$ added, and $u$ is the utility
function.

With $w(k) = \frac{1}{n} \binom{n-1}{k}^{-1}$, we recover the Shapley value.

Two additional instances of semi-value are [Data Banzhaf][data-banzhaf-intro]
[@wang_data_2023] and [Beta Shapley][beta-shapley-intro] [@kwon_beta_2022],
which offer improved numerical and rank stability in certain situations.

All semi-values, including those two, are implemented in pyDVL by composing
different sampling methods and weighting schemes. The abstract class from which
they derive is
[SemiValueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation],
whose main abstract method is the weighting scheme $k \mapsto w(k)$.


## General semi-values

In pyDVL we provide a general method for computing general semi-values with any
combination of the three ingredients that define them:

- A utility function $u$.
- A sampling method.
- A weighting scheme $w$.

You can construct any combination of these three ingredients with subclasses of
[SemivalueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation] and
any of the samplers defined in [pydvl.valuation.samplers][].

Allowing any combination enables testing different importance-sampling schemes
and can help when experimenting with models that are more sensitive to changes
in training set size.[^bzf-stability]

For more on this topic and how Monte Carlo sampling interacts with the
semi-value coefficient and the sampler probabilities, see [[semi-values-sampling]].


[^bzf-stability]: Note however that Data Banzhaf has shown to be among the most
    robust to variance in the utility function, in the sense of rank stability,
    across a range of models and datasets [@wang_data_2023].
