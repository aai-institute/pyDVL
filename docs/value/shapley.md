---
title: Shapley value
---

## Shapley value

The Shapley method is an approach to compute data values originating in
cooperative game theory. Shapley values are a common way of assigning payoffs to
each participant in a cooperative game (i.e. one in which players can form
coalitions) in a way that ensures that certain axioms are fulfilled.

pyDVL implements several methods for the computation and approximation of
Shapley values. They can all be accessed via the facade function
[compute_shapley_values][pydvl.value.shapley.compute_shapley_values].
The supported methods are enumerated in
[ShapleyMode][pydvl.value.shapley.ShapleyMode].

Empirically, the most useful method is the so-called *Truncated Monte Carlo
Shapley* [@ghorbani_data_2019], which is a Monte Carlo approximation of the
[permutation Shapley value][permutation-shapley].


### Combinatorial Shapley

The first algorithm is just a verbatim implementation of the definition. As such
it returns as exact a value as the utility function allows (see what this means
in [Problems of Data Values][problems-of-data-values]).

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
as a weighted sum of its marginal utility wrt. every possible coalition of
training samples within the training set:

$$
v(i) = \frac{1}{n} \sum_{S \subseteq D_{-i}}
\binom{n-1}{ | S | }^{-1} [u(S_{+i}) − u(S)]
,$$

where $D_{-i}$ denotes the set of samples in $D$ excluding $x_i$, and $S_{+i}$
denotes the set $S$ with $x_i$ added.

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

Because the number of subsets $S \subseteq D_{-i}$ is
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

### Class-wise Shapley

Class-wise Shapley [@schoch_csshapley_2022] offers a distinct Shapley framework tailored
for classification problems. Let $D$ be the dataset, $D_{y_i}$ be the subset of $D$ with
labels $y_i$, and $D_{-y_i}$ be the complement of $D_{y_i}$ in $D$. The key idea is that
a sample $(x_i, y_i)$, might enhance the overall performance on $D$, while being 
detrimental for the performance on $D_{y_i}$. To address this issue, the
authors introduced the estimator

$$
v_u(i) = \frac{1}{2^{|D_{-y_i}|}} \sum_{S_{-y_i}} \frac{1}{|D_{y_i}|!}
\sum_{S_{y_i}} \binom{|D_{y_i}|-1}{|S_{y_i}|}^{-1}
[u( S_{y_i} \cup \{i\} | S_{-y_i} ) − u( S_{y_i} | S_{-y_i})],
$$

where $S_{y_i} \subseteq D_{y_i} \setminus \{i\}$ and $S_{-y_i} \subseteq D_{-y_i}$. In
other words, the summations are over the powerset of $D_{y_i} \setminus \{i\}$ and 
$D_{-y_i}$ respectively. The algorithm can be applied by using the snippet

```python
from pydvl.utils import Dataset, Utility
from pydvl.value import HistoryDeviation, MaxChecks, RelativeTruncation
from pydvl.value.shapley.classwise import compute_classwise_shapley_values, \
    ClasswiseScorer

model = ...
data = Dataset(...)
scoring = ("accuracy")
utility = Utility(model, data, scoring)
values = compute_classwise_shapley_values(
    utility,
    done=HistoryDeviation(n_steps=500, rtol=5e-2),
    truncation=RelativeTruncation(utility, rtol=0.01),
    done_sample_complements=MaxChecks(1),
    normalize_values=True
)
```

where `ClasswiseScorer` is a special type of scorer only applicable for classification
problems. In practical applications, the evaluation of this estimator leverages both
Monte Carlo sampling and permutation Monte Carlo sampling [@castro_polynomial_2009].

### Owen sampling

**Owen Sampling** [@okhrati_multilinear_2021] is a practical algorithm based on
the combinatorial definition. It uses a continuous extension of the utility from
$\{0,1\}^n$, where a 1 in position $i$ means that sample $x_i$ is used to train
the model, to $[0,1]^n$. The ensuing expression for Shapley value uses
integration instead of discrete weights:

$$
v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{-i})} [u(S_{+i}) - u(S)].
$$

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
[StoppingCriterion][pydvl.value.stopping.StoppingCriterion] to the function, but
instead the number of iterations and the maximum number of samples to use in the
integration.

### Permutation Shapley

An equivalent way of computing Shapley values (`ApproShapley`) appeared in
[@castro_polynomial_2009] and is the basis for the method most often used in
practice. It uses permutations over indices instead of subsets:

$$
v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)}
[u(\sigma_{:i} \cup \{x_i\}) − u(\sigma_{:i})],
$$

where $\sigma_{:i}$ denotes the set of indices in permutation sigma before the
position where $i$ appears. To approximate this sum (which has $\mathcal{O}(n!)$
terms!) one uses Monte Carlo sampling of permutations, something which has
surprisingly low sample complexity. One notable difference wrt. the
combinatorial approach above is that the approximations always fulfill the
efficiency axiom of Shapley, namely $\sum_{i=1}^n \hat{v}_i = u(D)$ (see
[@castro_polynomial_2009], Proposition 3.2).

By adding two types of early stopping, the result is the so-called **Truncated
Monte Carlo Shapley** [@ghorbani_data_2019], which is efficient enough to be
useful in applications. The first is simply a convergence criterion, of which
there are [several to choose from][pydvl.value.stopping]. The second is a
criterion to truncate the iteration over single permutations.
[RelativeTruncation][pydvl.value.shapley.truncated.RelativeTruncation] chooses
to stop iterating over samples in a permutation when the marginal utility
becomes too small.

```python
from pydvl.value import compute_shapley_values, MaxUpdates, RelativeTruncation

values = compute_shapley_values(
    u=utility,
    mode="permutation_montecarlo",
    done=MaxUpdates(1000),
    truncation=RelativeTruncation(utility, rtol=0.01)
)
```

You can see this method in action in
[this example](../../examples/shapley_basic_spotify/) using the Spotify dataset.

### Exact Shapley for KNN

It is possible to exploit the local structure of K-Nearest Neighbours to reduce
the amount of subsets to consider: because no sample besides the K closest
affects the score, most are irrelevant and it is possible to compute a value in
linear time. This method was introduced by [@jia_efficient_2019a], and can be
used in pyDVL with:

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

An alternative approach introduced in [@jia_efficient_2019a] first approximates
the differences of values with a Monte Carlo sum. With

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
    can be several orders of magnitude worse than with e.g. TMCS. Additionally,
    the CSP can sometimes turn out to be infeasible.

Usage follows the same pattern as every other Shapley method, but with the
addition of an `epsilon` parameter required for the solution of the CSP. It
should be the same value used to compute the minimum number of samples required.
This can be done with
[num_samples_eps_delta][pydvl.value.shapley.gt.num_samples_eps_delta], but
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
