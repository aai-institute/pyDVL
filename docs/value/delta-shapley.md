---
title: Delta-Shapley
alias: 
  name: delta-shapley-intro
  text: Introduction to Delta-Shapley
---

# $\delta$-Shapley { #delta-shapley-intro }

!!! warning "Experimental"
    As of v0.10.0, the $\delta$-Shapley value is an experimental feature. It has
    not been tested enough and is known to contain bugs. PRs welcome!

!!! tip "Practical applications"
    While we provide an implementation of the $\delta$-Shapley value for the
    sake of completeness, in practice, properly adjusting the constants required
    is often difficult, making it hard to use. If one still wishes to use
    stratified sampling, we recommend using subset size sampling strategies that
    don't require these constants, such as
    [PowerLawSampleSize][pydvl.valuation.samplers.stratified.PowerLawSampleSize].

**$\delta$-Shapley** is a semi-value that approximates the average marginal
contribution per subset size, truncated for sizes beyond a certain range.  It
was introduced in [@watson_accelerated_2023], and is available in pyDVL as
[DeltaShapleyValuation][pydvl.valuation.methods.delta_shapley.DeltaShapleyValuation].

Let's decompose the definition of Shapley value into "layers", one per subset
size $k$, by letting $v_\text{shap}(i) = \frac{1}{n} \sum_{k=0}^{n-1}
\phi_i^{k},$ where 

$$\phi_i^{k} := \binom{n-1}{k}^{-1} 
                \sum_{S \subseteq D_{-i}^{k}} [u(S_{+i}) - u(S)],$$

and $D_i^{(k)}$ is the complement  of $\{i\}$, $u$ is the utility and $S_{+i}$
is the set $S$ with the point $i$ added to it. Since there are $\binom{n-1}{k}$
sets of size $k$, each $\phi_i^{k}$ is the average marginal contribution of the
point $i$ to all sets of size $k$.

Therefore, one can estimate $v_\text{shap}(i)$ by approximating the
$\phi_i^{(k)}$ and then averaging those. This approximation consists of sampling
only a fraction $m_k$ of all the sets of size $k$ and averaging the marginal
contributions. One of the contributions of the paper is a careful choice of
$m_k$ (see [below][sampling-for-delta-shapley]).

Additionally, the authors argue that, for certain model classes, small
coalitions are good estimators of the data value, because adding more data tends
to have diminishing returns for many models. This motivates clipping $k$ outside
a given range, to come to the final definition of the $\delta$-Shapley
value:[^def]

$$
v_\text{del}(i) := \frac{1}{u - l + 1} \sum_{k=l}^{u} \frac{1}{m_k} \sum_{j=1}^{m_k}
[u(S^j_{+i}) - u(S^j)]
$$

where $l$ and $u$ are lower and upper bounds for $k$, the sets $S^j$ are sampled
uniformly at random from $D_{-i}^{k}$, and $m_k$ is the number of samples at
size $k.$


## Sampling for Delta-Shapley  { #sampling-for-delta-shapley }

In $\delta$-Shapley, subset sizes are sampled according to the probability
$m_k/m$ (even though the exact procedure can vary, e.g. iterate through all $k$
[deterministically][pydvl.valuation.samplers.stratified.FiniteSequentialSizeIteration]
or [at random][pydvl.valuation.samplers.stratified.RandomSizeIteration]). This
means that the probability of sampling a set of size $k$ is

$$p(S|k) = \binom{n-1}{k}^{-1} \frac{m_k}{m},$$

which is the implicit coefficient for the average marginal contribution, that
one must account for when sampling sets stochastically (see [Sampling strategies
for semi-values][semi-values-sampling]).

The choice of $m_k$ is guided by theoretical bounds derived from the uniform
stability properties of the learning algorithm. The authors derive bounds for
different classes of loss functions using concentration inequalities, with
Theorems 6 and 7 providing the choice of $m_k$ for the case of non-convex,
smooth Lipschitz models trained with SGD.[^sgd] This is the case that we
implement in
[DeltaShapleyNCSGDSampleSize][pydvl.valuation.samplers.stratified.DeltaShapleyNCSGDSampleSize],
but we will discuss below how to implement any choice of $m_k$.

### Powerset and permutation sampling

The original paper uses a standard powerset sampling approach, where the sets
$S^j$ are sampled uniformly at random from the powerset of $D_{-i}^{k}.$ We
provide this sampling method via
[StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler],
which can be configured with any of the classes inheriting
[SampleSizeStrategy][pydvl.valuation.samplers.stratified.SampleSizeStrategy].
These implement the $m_k,$ and the lower and upper bounds truncating $k.$

Alternatively, we provide an **experimental and approximate** permutation-based
approach which clips permutations and keeps track of the sizes of sets returned.
This reduces computation by at least a factor of 2, since the evaluation
strategy can reuse the previously computed utility for the marginal
contribution. This is implemented in
[StratifiedPermutationSampler][pydvl.valuation.samplers.stratified.StratifiedPermutationSampler].
Note that it does not guarantee sampling the exact number of set sizes $m_k.$


## Delta-Shapley for non-convex SGD

Setting $m_k$ for a general model trained with SGD requires several parameters:
the number of SGD iterations, the range of the loss, the Lipschitz constant of
the model, and the learning rate, which is assumed to decay as $\alpha_t = c /
t.$ For the exact expressions see equations (8) and (9) of the paper.

All of these parameters must be set when instantiating the
[$\delta$-Shapley][pydvl.valuation.methods.delta_shapley.DeltaShapleyValuation]
class.

??? error "Inconsistencies between the paper and the code"
    There are several inconsistencies between the paper and the code that we
    could [find online](https://github.com/laurenwatson/delta-shapley), which
    we couldn't resolve to our satisfaction. These include:
    1. There are minor discrepancies in the definition of $m_k,$ most notably
       the introduction of a factor $n_\text{test}$.
    2. The code uses a certain, rather arbitrary, number of SGD iterations $T$
       to compute $m_k$ which is never actually used to train the model.
    3. Most constants are set to arbitrary values, seemingly without any
       justification, potentially invalidating the application of the
       theoretical bounds.
    For these reasons, we provide two modes of operation for the sample size
    strategy implementing these bounds to either follow those in the paper or
    those in the code, for reproducibility purposes. See
    [DeltaShapleyNCSGDSampleSize][pydvl.valuation.samplers.stratified.DeltaShapleyNCSGDSampleSize].


[^sgd]: A technical detail is the assumption that the order in which batches
    are sampled from a coalition $S$ when computing $u(s)$ is not random (i.e.
    it is constant across epochs).

[^def]: We believe Definition 9 in the paper to be a bit misleading since it
    lacks the averaging of the marginals over the sampled sets. As it stands,
    it iss an _unweighted_ average of the marginals, which would explode in
    magnitude for large $k.$ This seems substantiated by the fact that the code
    we found online does not implement this definition, but rather the one we
    provide here.
