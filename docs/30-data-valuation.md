---
title: Computing Data Values
alias: 
  name: data-valuation
  text: Computing Data Values
---

# Computing Data Values

**Data valuation** is the task of assigning a number to each element of a
training set which reflects its contribution to the final performance of a
model trained on it. This value is not an intrinsic property of the element of
interest, but a function of three factors:

1. The dataset $D$, or more generally, the distribution it was sampled
   from (with this we mean that *value* would ideally be the (expected)
   contribution of a data point to any random set $D$ sampled from the same
   distribution).

2. The algorithm $\mathcal{A}$ mapping the data $D$ to some estimator $f$
   in a model class $\mathcal{F}$. E.g. MSE minimization to find the parameters
   of a linear model.

3. The performance metric of interest $u$ for the problem. E.g. the $R^2$
   score or the negative MSE over a test set.

pyDVL collects algorithms for the computation of data values in this sense,
mostly those derived from cooperative game theory. The methods can be found in
the package [pydvl.value][pydvl.value] , with support from modules
[pydvl.utils.dataset][pydvl.utils.dataset]
and [pydvl.utils.utility][pydvl.utils.utility], as detailed below.

!!! Warning
    Be sure to read the section on
    [the difficulties using data values][problems-of-data-values].

## Creating a Dataset

The first item in the tuple $(D, \mathcal{A}, u)$ characterising data value is
the dataset. The class [Dataset][pydvl.utils.dataset.Dataset] is a simple
convenience wrapper for the train and test splits that is used throughout pyDVL.
The test set will be used to evaluate a scoring function for the model.

It can be used as follows:

```python
import numpy as np
from pydvl.utils import Dataset
from sklearn.model_selection import train_test_split
X, y = np.arange(100).reshape((50, 2)), np.arange(50)
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.5, random_state=16
)
dataset = Dataset(X_train, X_test, y_train, y_test)
```

It is also possible to construct Datasets from sklearn toy datasets for
illustrative purposes using [from_sklearn][pydvl.utils.dataset.Dataset.from_sklearn].

### Grouping data

Be it because data valuation methods are computationally very expensive, or
because we are interested in the groups themselves, it can be often useful or
necessary to group samples to valuate them together.
[GroupedDataset][pydvl.utils.dataset.GroupedDataset] provides an alternative to
[Dataset][pydvl.utils.dataset.Dataset] with the same interface which allows this.

You can see an example in action in the
[Spotify notebook](../examples/shapley_basic_spotify), but here's a simple
example grouping a pre-existing `Dataset`. First we construct an array mapping
each index in the dataset to a group, then use
[from_dataset][pydvl.utils.dataset.GroupedDataset.from_dataset]:

```python
import numpy as np
from pydvl.utils import GroupedDataset

# Randomly assign elements to any one of num_groups:
data_groups = np.random.randint(0, num_groups, len(dataset))
grouped_dataset = GroupedDataset.from_dataset(dataset, data_groups)
```

## Creating a Utility

In pyDVL we have slightly overloaded the name "utility" and use it to refer to
an object that keeps track of all three items in $(D, \mathcal{A}, u)$. This
will be an instance of [Utility][pydvl.utils.utility.Utility] which, as mentioned,
is a convenient wrapper for the dataset, model and scoring function used for
valuation methods.

Here's a minimal example:

```python
import sklearn as sk
from pydvl.utils import Dataset, Utility

dataset = Dataset.from_sklearn(sk.datasets.load_iris())
model = sk.svm.SVC()
utility = Utility(model, dataset)
```

The object `utility` is a callable that data valuation methods will execute
with different subsets of training data. Each call will retrain the model on a
subset and evaluate it on the test data using a scoring function. By default,
[Utility][pydvl.utils.utility.Utility] will use `model.score()`, but it is
possible to use any scoring function (greater values must be better). In
particular, the constructor accepts the same types as argument as sklearn's
[cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>):
a string, a scorer callable or [None][] for the default.

```python
utility = Utility(model, dataset, "explained_variance")
```

`Utility` will wrap the `fit()` method of the model to cache its results. This
greatly reduces computation times of Monte Carlo methods. Because of how caching
is implemented, it is important not to reuse `Utility` objects for different
datasets. You can read more about [setting up the cache][setting-up-the-cache]
in the installation guide and the documentation
of the [caching][pydvl.utils.caching] module.

### Using custom scorers

The `scoring` argument of [Utility][pydvl.utils.utility.Utility] can be used to
specify a custom [Scorer][pydvl.utils.utility.Scorer] object. This is a simple
wrapper for a callable that takes a model, and test data and returns a score.

More importantly, the object provides information about the range of the score,
which is used by some methods by estimate the number of samples necessary, and
about what default value to use when the model fails to train.

!!! Note
    The most important property of a `Scorer` is its default value. Because many
    models will fail to fit on small subsets of the data, it is important to
    provide a sensible default value for the score.

It is possible to skip the construction of the [Scorer][pydvl.utils.utility.Scorer]
when constructing the `Utility` object. The two following calls are equivalent:

```python
utility = Utility(
   model, dataset, "explained_variance", score_range=(-np.inf, 1), default_score=0.0
)
utility = Utility(
   model, dataset, Scorer("explained_variance", range=(-np.inf, 1), default=0.0)
)
```

### Learning the utility

Because each evaluation of the utility entails a full retrain of the model with
a new subset of the training set, it is natural to try to learn this mapping
from subsets to scores. This is the idea behind **Data Utility Learning (DUL)**
[@wang_improving_2022] and in pyDVL it's as simple as wrapping the
`Utility` inside [DataUtilityLearning][pydvl.utils.utility.DataUtilityLearning]:

```python
from pydvl.utils import Utility, DataUtilityLearning, Dataset
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris

dataset = Dataset.from_sklearn(load_iris())
u = Utility(LogisticRegression(), dataset, enable_cache=False)
training_budget = 3
wrapped_u = DataUtilityLearning(u, training_budget, LinearRegression())

# First 3 calls will be computed normally
for i in range(training_budget):
   _ = wrapped_u((i,))
# Subsequent calls will be computed using the fit model for DUL
wrapped_u((1, 2, 3))
```

As you can see, all that is required is a model to learn the utility itself and
the fitting and using of the learned model happens behind the scenes.

There is a longer example with an investigation of the results achieved by DUL
in [a dedicated notebook](../examples/shapley_utility_learning).

## Leave-One-Out values

The Leave-One-Out method is a simple approach that assigns each sample its
*marginal utility* as value:

$$v_u(x_i) = u(D) − u(D \setminus \{x_i\}).$$

For the purposes of data valuation, this is rarely useful beyond serving as a
baseline for benchmarking. One particular weakness is that it does not
necessarily correlate with an intrinsic value of a sample: since it is a
marginal utility, it is affected by the "law" of diminishing returns. Often, the
training set is large enough for a single sample not to have any significant
effect on training performance, despite any qualities it may possess. Whether
this is indicative of low value or not depends on each one's goals and
definitions, but other methods are typically preferable.

```python
from pydvl.value.loo.naive import naive_loo

values = naive_loo(utility)
```

The return value of all valuation functions is an object of type
[ValuationResult][pydvl.value.result.ValuationResult]. This can be iterated over,
indexed with integers, slices and Iterables, as well as converted to a
[pandas.DataFrame][].

## Shapley values

The Shapley method is an approach to compute data values originating in
cooperative game theory. Shapley values are a common way of assigning payoffs to
each participant in a cooperative game (i.e. one in which players can form
coalitions) in a way that ensures that certain axioms are fulfilled.

pyDVL implements several methods for the computation and approximation of
Shapley values. They can all be accessed via the facade function
[compute_shapley_values][pydvl.value.shapley.compute_shapley_values].
The supported methods are enumerated in
[ShapleyMode][pydvl.value.shapley.ShapleyMode].


### Combinatorial Shapley

The first algorithm is just a verbatim implementation of the definition. As such
it returns as exact a value as the utility function allows (see what this means
in Problems of Data Values][problems-of-data-values]).

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
as a weighted sum of its marginal utility wrt. every possible coalition of
training samples within the training set:

$$
v_u(x_i) = \frac{1}{n} \sum_{S \subseteq D \setminus \{x_i\}}
\binom{n-1}{ | S | }^{-1} [u(S \cup \{x_i\}) − u(S)]
,$$

```python
from pydvl.value import compute_shapley_values

values = compute_shapley_values(utility, mode="combinatorial_exact")
df = values.to_dataframe(column='value')
```

We can convert the return value to a
[pandas.DataFrame][].
and name the column with the results as `value`. Please refer to the
documentation in [shapley][pydvl.value.shapley] and
[ValuationResult][pydvl.value.result.ValuationResult] for more information.

### Monte Carlo Combinatorial Shapley

Because the number of subsets $S \subseteq D \setminus \{x_i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations. The simplest
one is done via Monte Carlo sampling of the powerset $\mathcal{P}(D)$. In pyDVL
this simple technique is called "Monte Carlo Combinatorial". The method has very
poor converge rate and others are preferred, but if desired, usage follows the
same pattern:

```python
from pydvl.value import compute_shapley_values, MaxUpdates

values = compute_shapley_values(
   utility, mode="combinatorial_montecarlo", done=MaxUpdates(1000)
)
df = values.to_dataframe(column='cmc')
```

The DataFrames returned by most Monte Carlo methods will contain approximate
standard errors as an additional column, in this case named `cmc_stderr`.

Note the usage of the object [MaxUpdates][pydvl.value.stopping.MaxUpdates] as the
stop condition. This is an instance of a
[StoppingCriterion][pydvl.value.stopping.StoppingCriterion]. Other examples are
[MaxTime][pydvl.value.stopping.MaxTime] and
[AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError].


### Owen sampling

**Owen Sampling** [@okhrati_multilinear_2021] is a practical
algorithm based on the combinatorial definition. It uses a continuous extension
of the utility from $\{0,1\}^n$, where a 1 in position $i$ means that sample
$x_i$ is used to train the model, to $[0,1]^n$. The ensuing expression for
Shapley value uses integration instead of discrete weights:

$$
v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{\backslash \{ i \}})}
[u(S \cup {i}) - u(S)]
.$$

Using Owen sampling follows the same pattern as every other method for Shapley
values in pyDVL. First construct the dataset and utility, then call
[compute_shapley_values][pydvl.value.shapley.compute_shapley_values]:

```python
from pydvl.value import compute_shapley_values

values = compute_shapley_values(
   u=utility, mode="owen", n_iterations=4, max_q=200
)
```

There are more details on Owen sampling, and its variant *Antithetic Owen
Sampling* in the documentation for the function doing the work behind the scenes:
[owen_sampling_shapley][pydvl.value.shapley.owen.owen_sampling_shapley].

Note that in this case we do not pass a
[StoppingCriterion][pydvl.value.stopping.StoppingCriterion] to the function, but instead
the number of iterations and the maximum number of samples to use in the
integration.

### Permutation Shapley

An equivalent way of computing Shapley values (`ApproShapley`) appeared in
[@castro_polynomial_2009] and is the basis for the method most often
used in practice. It uses permutations over indices instead of subsets:

$$
v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
[u(\sigma_{:i} \cup \{i\}) − u(\sigma_{:i})]
,$$

where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
position where $i$ appears. To approximate this sum (which has $\mathcal{O}(n!)$
terms!) one uses Monte Carlo sampling of permutations, something which has
surprisingly low sample complexity. One notable difference wrt. the
combinatorial approach above is that the approximations always fulfill the
efficiency axiom of Shapley, namely $\sum_{i=1}^n \hat{v}_i = u(D)$ (see
[@castro_polynomial_2009], Proposition 3.2).

By adding early stopping, the result is the so-called **Truncated Monte Carlo
Shapley** [@ghorbani_data_2019], which is efficient enough to be
useful in applications.

```python
from pydvl.value import compute_shapley_values, MaxUpdates

values = compute_shapley_values(
   u=utility, mode="truncated_montecarlo", done=MaxUpdates(1000)
)
```


### Exact Shapley for KNN

It is possible to exploit the local structure of K-Nearest Neighbours to reduce
the amount of subsets to consider: because no sample besides the K closest
affects the score, most are irrelevant and it is possible to compute a value in
linear time. This method was introduced by [@jia_efficient_2019a],
and can be used in pyDVL with:

```python
from pydvl.utils import Dataset, Utility
from pydvl.value import compute_shapley_values
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
data = Dataset(...)
utility = Utility(model, data)
values = compute_shapley_values(u=utility, mode="knn")
```

### Group testing

An alternative approach introduced in [@jia_efficient_2019a]
first approximates the differences of values with a Monte Carlo sum. With

$$\hat{\Delta}_{i j} \approx v_i - v_j,$$

one then solves the following linear constraint satisfaction problem (CSP) to
infer the final values:

$$
\begin{array}{lll}
\sum_{i = 1}^N v_i & = & U (D)\\
| v_i - v_j - \hat{\Delta}_{i j} | & \leqslant &
\frac{\varepsilon}{2 \sqrt{N}}
\end{array}
$$

!!! Warning
    We have reproduced this method in pyDVL for completeness and benchmarking,
    but we don't advocate its use because of the speed and memory cost. Despite
    our best efforts, the number of samples required in practice for convergence
    can be several orders of magnitude worse than with e.g. Truncated Monte Carlo.
    Additionally, the CSP can sometimes turn out to be infeasible.

Usage follows the same pattern as every other Shapley method, but with the
addition of an `epsilon` parameter required for the solution of the CSP. It
should be the same value used to compute the minimum number of samples required.
This can be done with [num_samples_eps_delta][pydvl.value.shapley.gt.num_samples_eps_delta], but
note that the number returned will be huge! In practice, fewer samples can be
enough, but the actual number will strongly depend on the utility, in particular
its variance.

```python
from pydvl.utils import Dataset, Utility
from pydvl.value import compute_shapley_values

model = ...
data = Dataset(...)
utility = Utility(model, data, score_range=(_min, _max))
min_iterations = num_samples_eps_delta(epsilon, delta, n, utility.score_range)
values = compute_shapley_values(
   u=utility, mode="group_testing", n_iterations=min_iterations, eps=eps
)
```

## Core values

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
  $$\sum_{x_i\in S} v_u(x_i) \geq u(S), \forall S \subset D\,$$

The second property states that the sum of payoffs to the agents
in any subcoalition $S$ is at least as large as the amount that
these agents could earn by forming a coalition on their own.

### Least Core values

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

### Exact Least Core

This first algorithm is just a verbatim implementation of the definition.
As such it returns as exact a value as the utility function allows
(see what this means in Problems of Data Values][problems-of-data-values]).

```python
from pydvl.value import compute_least_core_values

values = compute_least_core_values(utility, mode="exact")
```

### Monte Carlo Least Core

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


## Semi-values

Shapley values are a particular case of a more general concept called semi-value,
which is a generalization to different weighting schemes. A **semi-value** is
any valuation function with the form:

$$
v\_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D\_{-i}^{(k)}} [U(S\_{+i})-U(S)],
$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

Two instances of this are **Banzhaf indices** [@wang_data_2022],
and **Beta Shapley** [@kwon_beta_2022], with better numerical and
rank stability in certain situations.

!!! Note
    Shapley values are a particular case of semi-values and can therefore also be
    computed with the methods described here. However, as of version 0.6.0, we
    recommend using [compute_shapley_values][pydvl.value.shapley.compute_shapley_values] instead,
    in particular because it implements truncated Monte Carlo sampling for faster
    computation.


### Beta Shapley

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
from pydvl.value import compute_semivalues

values = compute_semivalues(
   u=utility, mode="beta_shapley", done=MaxUpdates(500), alpha=1, beta=16
)
```

### Banzhaf indices

As noted below in the [Problems of Data Values][problems-of-data-values] section,
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
from pydvl.value import compute_semivalues, MaxUpdates

values = compute_semivalues( u=utility, mode="banzhaf", done=MaxUpdates(500))
```

## Problems of data values

There are a number of factors that affect how useful values can be for your
project. In particular, regression can be especially tricky, but the particular
nature of every (non-trivial) ML problem can have an effect:

* **Unbounded utility**: Choosing a scorer for a classifier is simple: accuracy
  or some F-score provides a bounded number with a clear interpretation. However,
  in regression problems most scores, like $R^2$, are not bounded because
  regressors can be arbitrarily bad. This leads to great variability in the
  utility for low sample sizes, and hence unreliable Monte Carlo approximations
  to the values. Nevertheless, in practice it is only the ranking of samples
  that matters, and this tends to be accurate (wrt. to the true ranking) despite
  inaccurate values.

  pyDVL offers a dedicated [function composition][pydvl.utils.score.compose_score]
  for scorer functions which can be used to squash a score.
  The following is defined in module [score][pydvl.utils.score]:

  ```python
  import numpy as np
  from pydvl.utils.types import compose_score
  
  def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))
  
  squashed_r2 = compose_score("r2", sigmoid, "squashed r2")
  
  squashed_variance = compose_score(
    "explained_variance", sigmoid, "squashed explained variance"
  )
  ```

  These squashed scores can prove useful in regression problems, but they can
  also introduce issues in the low-value regime.

* **High variance utility**: Classical applications of game theoretic value
  concepts operate with deterministic utilities, but in ML we use an evaluation
  of the model on a validation set as a proxy for the true risk. Even if the
  utility *is* bounded, if it has high variance then values will also have high
  variance, as will their Monte Carlo estimates. One workaround in pyDVL is to
  configure the caching system to allow multiple evaluations of the utility for
  every index set. A moving average is computed and returned once the standard
  error is small, see [MemcachedConfig][pydvl.utils.config.MemcachedConfig].

  [@wang_data_2022] prove that by relaxing one of the Shapley axioms
  and considering the general class of semi-values, of which Shapley is an
  instance, one can prove that a choice of constant weights is the best one can
  do in a utility-agnostic setting. So-called *Data Banzhaf* is on our to-do
  list!

* **Data set size**: Computing exact Shapley values is NP-hard, and Monte Carlo
  approximations can converge slowly. Massive datasets are thus impractical, at
  least with current techniques. A workaround is to group samples and investigate
  their value together. In pyDVL you can do this using
  [GroupedDataset][pydvl.utils.dataset.GroupedDataset]. 
  There is a fully worked-out [example here](../examples/shapley_basic_spotify).
  Some algorithms also provide different sampling strategies to reduce 
  the variance, but due to a no-free-lunch-type theorem,
  no single strategy can be optimal for all utilities.

* **Model size**: Since every evaluation of the utility entails retraining the
  whole model on a subset of the data, large models require great amounts of
  computation. But also, they will effortlessly interpolate small to medium
  datasets, leading to great variance in the evaluation of performance on the
  dedicated validation set. One mitigation for this problem is cross-validation,
  but this would incur massive computational cost. As of v.0.3.0 there are no
  facilities in pyDVL for cross-validating the utility (note that this would
  require cross-validating the whole value computation).
