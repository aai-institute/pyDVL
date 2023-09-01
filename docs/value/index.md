---
title: Data valuation
alias: 
  name: data-valuation
  text: Basics of data valuation
---

# Data valuation

!!! Note
    If you want to jump right into the steps to compute values, skip ahead
    to [Computing data values](#computing-data-values).

**Data valuation** is the task of assigning a number to each element of a
training set which reflects its contribution to the final performance of some
model trained on it. Some methods attempt to be model-agnostic, but in most
cases the model is an integral part of the method. In these cases, this number
not an intrinsic property of the element of interest, but typically a function
of three factors:

1. The dataset $D$, or more generally, the distribution it was sampled
   from (with this we mean that *value* would ideally be the (expected)
   contribution of a data point to any random set $D$ sampled from the same
   distribution).

2. The algorithm $\mathcal{A}$ mapping the data $D$ to some estimator $f$
   in a model class $\mathcal{F}$. E.g. MSE minimization to find the parameters
   of a linear model.

3. The performance metric of interest $u$ for the problem. When value depends on
   a model, it must be measured in some way which uses it. E.g. the $R^2$ score or
   the negative MSE over a test set.

pyDVL collects algorithms for the computation of data values in this sense,
mostly those derived from cooperative game theory. The methods can be found in
the package [pydvl.value][pydvl.value] , with support from modules
[pydvl.utils.dataset][pydvl.utils.dataset]
and [pydvl.utils.utility][pydvl.utils.utility], as detailed below.

!!! Warning
    Be sure to read the section on
    [the difficulties using data values][problems-of-data-values].

There are three main families of methods for data valuation: game-theoretic, 
influence-based and intrinsic. As of v0.7.0 pyDVL supports the first two. Here,
we focus on game-theoretic concepts and refer to the main documentation on the
[influence funtion][the-influence-function] for the second.

## Game theoretical methods

The main contenders in game-theoretic approaches are [Shapley
values](shapley.md]) [@ghorbani_data_2019], [@kwon_efficient_2021],
[@schoch_csshapley_2022], their generalization to so-called
[semi-values](semi-values.md) by [@kwon_beta_2022] and [@wang_data_2022],
and [the Core](the-core.md) [@yan_if_2021]. All of these are implemented
in pyDVL.

In these methods, data points are considered players in a cooperative game 
whose outcome is the performance of the model when trained on subsets 
(*coalitions*) of the data, measured on a held-out **valuation set**. This 
outcome, or **utility**, must typically be computed for *every* subset of 
the training set, so that an exact computation is $\mathcal{O} (2^n)$ in the 
number of samples $n$, with each iteration requiring a full re-fitting of the 
model using a coalition as training set. Consequently, most methods involve 
Monte Carlo approximations, and sometimes approximate utilities which are 
faster to compute, e.g. proxy models [@wang_improving_2022] or constant-cost
approximations like Neural Tangent Kernels [@wu_davinz_2022].

The reasoning behind using game theory is that, in order to be useful, an
assignment of value, dubbed **valuation function**, is usually required to
fulfil certain requirements of consistency and "fairness". For instance, in some
applications value should not depend on the order in which data are considered,
or it should be equal for samples that contribute equally to any subset of the
data (of equal size). When considering aggregated value for (sub-)sets of data
there are additional desiderata, like having a value function that does not
increase with repeated samples. Game-theoretic methods are all rooted in axioms
that by construction ensure different desiderata, but despite their practical
usefulness, none of them are either necessary or sufficient for all
applications. For instance, SV methods try to equitably distribute all value
among all samples, failing to identify repeated ones as unnecessary, with e.g. a
zero value.


## Applications of data valuation

Many applications are touted for data valuation, but the results can be
inconsistent. Values have a strong dependency on the training procedure and the
performance metric used. For instance, accuracy is a poor metric for imbalanced
sets and this has a stark effect on data values. Some models exhibit great
variance in some regimes and this again has a detrimental effect on values.

Nevertheless, some of the most promising applications are: Cleaning of corrupted
data, pruning unnecessary or irrelevant data, repairing mislabeled data, guiding
data acquisition and annotation (active learning), anomaly detection and model
debugging and interpretation.

## Computing data values

Using pyDVL to compute data values is a simple process that can be broken down
into three steps:

1. Creating a [Dataset][pydvl.utils.dataset.Dataset] object from your data.
2. Creating a [Utility][pydvl.utils.utility.Utility] which ties your model to
   the dataset and a [scoring function][pydvl.utils.utility.Scorer].
3. Computing values with a method of your choice, e.g. via
   [compute_shapley_values][pydvl.value.shapley.common.compute_shapley_values].

### Creating a Dataset

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

#### Grouping data

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

### Creating a Utility

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

The object `utility` is a callable that data valuation methods will execute with
different subsets of training data. Each call will retrain the model on a subset
and evaluate it on the test data using a scoring function. By default,
[Utility][pydvl.utils.utility.Utility] will use `model.score()`, but it is
possible to use any scoring function (greater values must be better). In
particular, the constructor accepts the same types as argument as
[sklearn.model_selection.cross_validate][]: a string, a scorer callable or
[None][] for the default.

```python
utility = Utility(model, dataset, "explained_variance")
```

`Utility` will wrap the `fit()` method of the model to cache its results. This
greatly reduces computation times of Monte Carlo methods. Because of how caching
is implemented, it is important not to reuse `Utility` objects for different
datasets. You can read more about [setting up the cache][setting-up-the-cache]
in the installation guide and the documentation
of the [caching][pydvl.utils.caching] module.

#### Using custom scorers

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
from pydvl.utils import Utility, Scorer

utility = Utility(
   model, dataset, "explained_variance", score_range=(-np.inf, 1), default_score=0.0
)
utility = Utility(
   model, dataset, Scorer("explained_variance", range=(-np.inf, 1), default=0.0)
)
```

#### Learning the utility

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

### Leave-One-Out values

LOO is the simplest approach to valuation. It assigns to each sample its
*marginal utility* as value:

$$v_u(i) = u(D) − u(D_{-i}).$$

For notational simplicity, we consider the valuation function as defined over
the indices of the dataset $D$, and $i \in D$ is the index of the sample,
$D_{-i}$ is the training set without the sample $x_i$, and $u$ is the utility
function.

For the purposes of data valuation, this is rarely useful beyond serving as a
baseline for benchmarking. Although in some benchmarks it can perform
astonishingly well on occasion. One particular weakness is that it does not
necessarily correlate with an intrinsic value of a sample: since it is a
marginal utility, it is affected by diminishing returns. Often, the training set
is large enough for a single sample not to have any significant effect on
training performance, despite any qualities it may possess. Whether this is
indicative of low value or not depends on each one's goals and definitions, but
other methods are typically preferable.

```python
from pydvl.value.loo import compute_loo

values = compute_loo(utility, n_jobs=-1)
```

The return value of all valuation functions is an object of type
[ValuationResult][pydvl.value.result.ValuationResult]. This can be iterated over,
indexed with integers, slices and Iterables, as well as converted to a
[pandas.DataFrame][].


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
  from pydvl.utils import compose_score
  
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
  do in a utility-agnostic setting. So-called *Data Banzhaf*.

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
