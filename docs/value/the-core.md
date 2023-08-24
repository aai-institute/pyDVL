---
title: The Least Core for Data Valuation
---

# Core values

The Shapley values define a fair way to distribute payoffs amongst all
participants when they form a grand coalition. But they do not consider
the question of stability: under which conditions do all participants
form the grand coalition? Would the participants be willing to form
the grand coalition given how the payoffs are assigned,
or would some of them prefer to form smaller coalitions?

The Core is another approach to computing data values originating
in cooperative game theory that attempts to ensure this stability.
It is the set of feasible payoffs that cannot be improved upon
by a coalition of the participants.

It satisfies the following 2 properties:

- **Efficiency**:
  The payoffs are distributed such that it is not possible
  to make any participant better off
  without making another one worse off.
  $$\sum_{x_i\in D} v_u(x_i) = u(D)\,$$

- **Coalitional rationality**:
  The sum of payoffs to the agents in any coalition S is at
  least as large as the amount that these agents could earn by
  forming a coalition on their own.
  $$\sum_{x_i \in S} v_u(x_i) \geq u(S), \forall S \subset D\,$$

The second property states that the sum of payoffs to the agents
in any subcoalition $S$ is at least as large as the amount that
these agents could earn by forming a coalition on their own.

## Least Core values

Unfortunately, for many cooperative games the Core may be empty.
By relaxing the coalitional rationality property by a subsidy $e \gt 0$,
we are then able to find approximate payoffs:

$$
\sum_{x_i\in S} v_u(x_i) + e \geq u(S), \forall S \subset D, S \neq \emptyset \
,$$

The least core value $v$ of the $i$-th sample in dataset $D$ wrt.
utility $u$ is computed by solving the following Linear Program:

$$
\begin{array}{lll}
\text{minimize} & e & \\
\text{subject to} & \sum_{x_i\in D} v_u(x_i) = u(D) & \\
& \sum_{x_i\in S} v_u(x_i) + e \geq u(S) &, \forall S \subset D, S \neq \emptyset  \\
\end{array}
$$

## Exact Least Core

This first algorithm is just a verbatim implementation of the definition.
As such it returns as exact a value as the utility function allows
(see what this means in Problems of Data Values][problems-of-data-values]).

```python
from pydvl.value import compute_least_core_values

values = compute_least_core_values(utility, mode="exact")
```

## Monte Carlo Least Core

Because the number of subsets $S \subseteq D \setminus \{x_i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations.

The simplest approximation consists in using a fraction of all subsets for the
constraints. [@yan_if_2021] show that a quantity of order
$\mathcal{O}((n - \log \Delta ) / \delta^2)$ is enough to obtain a so-called
$\delta$-*approximate least core* with high probability. I.e. the following
property holds with probability $1-\Delta$ over the choice of subsets:

$$
\mathbb{P}_{S\sim D}\left[\sum_{x_i\in S} v_u(x_i) + e^{*} \geq u(S)\right]
\geq 1 - \delta,
$$

where $e^{*}$ is the optimal least core subsidy.

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
