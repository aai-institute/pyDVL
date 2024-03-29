---
title: The Least Core for Data Valuation
---

# Core values

Shapley values define a fair way to distribute payoffs amongst all participants
(training points) when they form a grand coalition, i.e. when the model is
trained on the whole dataset. But they do not consider the question of
stability: under which conditions do all participants in a game form the grand
coalition? Are the payoffs distributed in such a way that prioritizes its
formation?

The Core is another solution concept in cooperative game theory that attempts to
ensure stability in the sense that it provides the set of feasible payoffs that
cannot be improved upon by a sub-coalition. This can be interesting for some
applications of data valuation because it yields values consistent with training
on the whole dataset, avoiding the spurious selection of subsets.

It satisfies the following 2 properties:

- **Efficiency**:
  The payoffs are distributed such that it is not possible to make any
  participant better off without making another one worse off.
  $\sum_{i \in D} v(i) = u(D).$

- **Coalitional rationality**:
  The sum of payoffs to the agents in any coalition $S$ is at least as large as
  the amount that these agents could earn by forming a coalition on their own.
  $\sum_{i \in S} v(i) \geq u(S), \forall S \subset D.$

The Core was first introduced into data valuation by [@yan_if_2021], in the
following form.

## Least Core values

Unfortunately, for many cooperative games the Core may be empty. By relaxing the
coalitional rationality property by a subsidy $e \gt 0$, we are then able to
find approximate payoffs:

$$
\sum_{i\in S} v(i) + e \geq u(S), \forall S \subset D, S \neq \emptyset \
,$$

The Least Core (LC) values $\{v\}$ for utility $u$ are computed by solving the
following linear program:

$$
\begin{array}{lll}
\text{minimize} & e & \\
\text{subject to} & \sum_{i\in D} v(i) = u(D) & \\
& \sum_{i\in S} v(i) + e \geq u(S) &, \forall S \subset D, S \neq \emptyset  \\
\end{array}
$$

Note that solving this program yields a _set of solutions_ $\{v_j:N \rightarrow
\mathbb{R}\}$, whereas the Shapley value is a single function $v$. In order to
obtain a single valuation to use, one breaks ties by solving a quadratic program
to select the $v$ in the LC with the smallest $\ell_2$ norm. This is called the
_egalitarian least core_.

## Exact Least Core

This first algorithm is just a verbatim implementation of the definition, in
[compute_least_core_values][pydvl.value.least_core.compute_least_core_values].
It computes all constraints for the linear problem by evaluating the utility on
every subset of the training data, and returns as exact a value as the utility
function allows (see what this means in [Problems of Data
Values][problems-of-data-values]).

```python
from pydvl.value import compute_least_core_values

values = compute_least_core_values(utility, mode="exact")
```

## Monte Carlo Least Core

Because the number of subsets $S \subseteq D \setminus \{i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations.

The simplest one consists in using a fraction of all subsets for the constraints.
[@yan_if_2021] show that a quantity of order $\mathcal{O}((n - \log \Delta ) /
\delta^2)$ is enough to obtain a so-called $\delta$-*approximate least core*
with high probability. I.e. the following property holds with probability
$1-\Delta$ over the choice of subsets:

$$
\mathbb{P}_{S\sim D}\left[\sum_{i\in S} v(i) + e^{*} \geq u(S)\right]
\geq 1 - \delta,
$$

where $e^{*}$ is the optimal least core subsidy. This approximation is
also implemented in
[compute_least_core_values][pydvl.value.least_core.compute_least_core_values]:

```python
from pydvl.value import compute_least_core_values

values = compute_least_core_values(
   utility, mode="montecarlo", n_iterations=n_iterations
)
```

!!! Note
    Although any number is supported, it is best to choose `n_iterations` to be
    at least equal to the number of data points.

Because computing the Least Core values requires the solution of a linear and a
quadratic problem *after* computing all the utility values, we offer the
possibility of splitting the latter from the former. This is useful when running
multiple experiments: use
[mclc_prepare_problem][pydvl.value.least_core.montecarlo.mclc_prepare_problem] to prepare a
list of problems to solve, then solve them in parallel with
[lc_solve_problems][pydvl.value.least_core.common.lc_solve_problems].

```python
from pydvl.value.least_core import mclc_prepare_problem, lc_solve_problems

n_experiments = 10
problems = [mclc_prepare_problem(utility, n_iterations=n_iterations)
    for _ in range(n_experiments)]
values = lc_solve_problems(problems)
```

## Method comparison

The TransferLab team reproduced the results of the original paper in a
publication for the 2022 MLRC [@benmerzoug_re_2023].

![Best sample removal on binary image
classification](img/mclc-best-removal-10k-natural.svg){ align=left width=50% class=invertible}

Roughly speaking, MCLC performs better in identifying **high value** points, as
measured by best-sample removal tasks. In all other aspects, it performs worse
or similarly to TMCS at comparable sample budgets. But using an equal number of
subsets is more computationally expensive because of the need to solve large
linear and quadratic optimization problems.


![Worst sample removal on binary image
classification](img/mclc-worst-removal-10k-natural.svg){ align=right width=50% class=invertible}

For these reasons we recommend some variation of SV like TMCS for outlier
detection, data cleaning and pruning, and perhaps MCLC for the selection of
interesting points to be inspected for the improvement of data collection or
model design.
