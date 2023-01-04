.. _data valuation:

=====================
Computing data values
=====================

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
the package :mod:`~pydvl.value`, with support from modules
:mod:`pydvl.utils.dataset` and :mod:`~pydvl.utils.utility`, as detailed below.

.. warning::
   Be sure to read the section on
   :ref:`the difficulties using data values <problems of data values>`.

Creating a Dataset
==================

The first item in the tuple $(D, \mathcal{A}, u)$ characterising data value is
the dataset. The class :class:`~pydvl.utils.dataset.Dataset` is a simple
convenience wrapper for the train and test splits that is used throughout pyDVL.
The test set will be used to evaluate a scoring function for the model.

It can be used as follows:

.. code-block:: python

   import numpy as np
   from pydvl.utils import Dataset
   from sklearn.model_selection import train_test_split

   X, y = np.arange(100).reshape((50, 2)), np.arange(50)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=16
   )

   dataset = Dataset(X_train, X_test, y_train, y_test)

It is also possible to construct Datasets from sklearn toy datasets for
illustrative purposes using :meth:`~pydvl.utils.dataset.Dataset.from_sklearn`.

Grouping data
^^^^^^^^^^^^^

Be it because data valuation methods are computationally very expensive, or
because we are interested in the groups themselves, it can be often useful or
necessary to group samples so as to valuate them together.
:class:`~pydvl.utils.dataset.GroupedDataset` provides an alternative to
`Dataset` with the same interface which allows this.

You can see an example in action in the
:doc:`Spotify notebook <examples/shapley_basic_spotify>`, but here's a simple
example grouping a pre-existing `Dataset`. First we construct an array mapping
each index in the dataset to a group, then use
:meth:`~pydvl.utils.dataset.GroupedDataset.from_dataset`:

.. code-block:: python

   # Randomly assign elements to any one of num_groups:
   data_groups = np.random.randint(0, num_groups, len(dataset))
   grouped_dataset = GroupedDataset.from_dataset(dataset, data_groups)
   grouped_utility = Utility(model=model, data=grouped_dataset)

Creating a Utility
==================

In pyDVL we have slightly overloaded the name "utility" and use it to refer to
an object that keeps track of all three items in $(D, \mathcal{A}, u)$. This
will be an instance of :class:`~pydvl.utils.utility.Utility` which, as mentioned,
is a convenient wrapper for the dataset, model and scoring function used for
valuation methods.

Here's a minimal example:

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   import sklearn as sk

   dataset = Dataset.from_sklearn(sk.datasets.load_iris())
   model = sk.svm.SVC()
   utility = Utility(model, dataset)

The object `utility` is a callable that data valuation methods will execute
with different subsets of training data. Each call will retrain the model on a
subset and evaluate it on the test data using a scoring function. By default,
:class:`~pydvl.utils.utility.Utility` will use `model.score()`, but it is
possible to use any scoring function (greater values must be better). In
particular, the constructor accepts the same types as argument as sklearn's
`cross_validate() <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_:
a string, a scorer callable or `None` for the default.

.. code-block:: python

   utility = Utility(model, dataset, "explained_variance")


`Utility` will wrap the `fit()` method of the model to cache its results. This
greatly reduces computation times of Monte Carlo methods. Because of how caching
is implemented, it is important not to reuse `Utility` objects for different
datasets. You can read more about :ref:`caching setup` in the installation guide
and the documentation of the :mod:`pydvl.utils.caching` module.

Learning the utility
^^^^^^^^^^^^^^^^^^^^

Because each evaluation of the utility entails a full retrain of the model with
a new subset of the training set, it is natural to try to learn this mapping
from subsets to scores. This is the idea behind **Data Utility Learning (DUL)**
(:footcite:t:`wang_improving_2022`) and in pyDVL it's as simple as wrapping the
`Utility` inside :class:`~pydvl.utils.utility.DataUtilityLearning`:

.. code-block::python

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

As you can see, all that is required is a model to learn the utility itself and
the fitting and using of the learned model happens behind the scenes.

There is a longer example with an investigation of the results achieved by DUL
in :doc:`a dedicated notebook <examples/shapley_utility_learning>`.

.. _LOO:

Leave-One-Out values
====================

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

.. code-block:: python

   from pydvl.value.loo.naive import naive_loo
   utility = Utility(...)
   values = naive_loo(utility)

The return value of all valuation functions is an object of type
:class:`~pydvl.value.results.ValuationResult`. This can be iterated over,
indexed with integers, slices and Iterables, as well as converted to a
`pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. _Shapley:

Shapley values
==============

The Shapley method is an approach to compute data values originating in
cooperative game theory. Shapley values are a common way of assigning payoffs to
each participant in a cooperative game (i.e. one in which players can form
coalitions) in a way that ensures that certain axioms are fulfilled.

pyDVL implements several methods for the computation and approximation of
Shapley values. They can all be accessed via the facade function
:func:`~pydvl.value.shapley.compute_shapley_values`. The supported methods are
enumerated in :class:`~pydvl.value.shapley.ShapleyMode`.


Combinatorial Shapley
^^^^^^^^^^^^^^^^^^^^^

The first algorithm is just a verbatim implementation of the definition. As such
it returns as exact a value as the utility function allows (see what this means
in :ref:`problems of data values`).

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
as a weighted sum of its marginal utility wrt. every possible coalition of
training samples within the training set:

$$v_u(x_i) = \frac{1}{n} \sum_{S \subseteq D \setminus \{x_i\}} \binom{n-1}{ | S | }^{-1} [u(S \cup \{x_i\}) − u(S)] ,$$

.. code-block:: python

   from pydvl.value import compute_shapley_value
   utility = Utility(...)
   values = compute_shapley_values(utility, mode="combinatorial_exact")
   df = values.to_dataframe(column='value')

We convert the return value to a
`pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
and name the column with the results as `value`. Please refer to the
documentation in :mod:`pydvl.value.shapley` and
:class:`~pydvl.value.results.ValuationResult` for more information.

Monte Carlo Combinatorial Shapley
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the number of subsets $S \subseteq D \setminus \{x_i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations. The simplest
one is done via Monte Carlo sampling of the powerset $\mathcal{P}(D)$. In pyDVL
this simple technique is called "Monte Carlo Combinatorial". The method has very
poor converge rate and others are preferred, but if desired, usage follows the
same pattern:

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.shapley import compute_shapley_values
   model = ...
   data = Dataset(...)
   utility = Utility(model, data)
   values = compute_shapley_values(utility, mode="combinatorial_montecarlo")
   df = values.to_dataframe(column='cmc')

The DataFrames returned by most Monte Carlo methods will contain approximate
standard errors as an additional column, in this case named `cmc_stderr`.


Owen sampling
^^^^^^^^^^^^^

**Owen Sampling** (:footcite:t:`okhrati_multilinear_2021`) is a practical
algorithm based on the combinatorial definition. It uses a continuous extension
of the utility from $\{0,1\}^n$, where a 1 in position $i$ means that sample
$x_i$ is used to train the model, to $[0,1]^n$. The ensuing expression for
Shapley value uses integration instead of discrete weights:

$$v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{\backslash \{ i \}})} [u(S \cup {i}) - u(S)].$$

Using Owen sampling follows the same pattern as every other method for Shapley
values in pyDVL. First construct the dataset and utility, then call
:func:`~pydvl.value.shapley.compute_shapley_values`:

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.shapley import compute_shapley_values
   model = ...
   dataset = Dataset(...)
   utility = Utility(data, model)
   values = compute_shapley_values(
       u=utility, mode="owen", max_iterations=4, max_q=200
   )

There are more details on Owen
sampling, and its variant *Antithetic Owen Sampling* in the documentation for the
function doing the work behind the scenes:
:func:`~pydvl.value.shapley.montecarlo.owen_sampling_shapley`.


Permutation Shapley
^^^^^^^^^^^^^^^^^^^

An equivalent way of computing Shapley values appears often in the literature.
It uses permutations over indices instead of subsets:

$$v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)} [u(\sigma_{i-1} \cup {i}) − u(\sigma_{i})],$$

where $\sigma_i$ denotes the set of indices in permutation sigma up until the
position of index $i$. To approximate this sum (with $\mathcal{O}(n!)$ terms!)
one uses Monte Carlo sampling of permutations, something which has surprisingly
low sample complexity. By adding early stopping, the result is the so-called
**Truncated Monte Carlo Shapley** (:footcite:t:`ghorbani_data_2019`), which is
efficient enough to be useful in some applications.

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.shapley import compute_shapley_values

   model = ...
   data = Dataset(...)
   utility = Utility(model, data)
   values = compute_shapley_values(
       u=utility, mode="truncated_montecarlo", max_iterations=100
   )


Exact Shapley for KNN
^^^^^^^^^^^^^^^^^^^^^

It is possible to exploit the local structure of K-Nearest Neighbours to reduce
the amount of subsets to consider: because no sample besides the K closest
affects the score, most are irrelevant and it is possible to compute a value in
linear time. This method was introduced by :footcite:t:`jia_efficient_2019a`,
and can be used in pyDVL with:

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.shapley import compute_shapley_values
   from sklearn.neighbors import KNeighborsClassifier

   model = KNeighborsClassifier(n_neighbors=5)
   data = Dataset(...)
   utility = Utility(model, data)
   values = compute_shapley_values(u=utility, mode="knn")

.. _Least Core:

Core values
===========

The Shapley values define a fair way to distribute payoffs amongst all participants when they form a grand coalition.
But they do not consider the question of stability: under which conditions do all participants form the grand coalition?
Would the participants be willing to form the grand coalition given how the payoffs are assigned,
or would some of them prefer to form smaller coalitions?

The Core is another approach to computing data values originating
in cooperative game theory that attempts to ensure this stability.
It is the set of feasible payoffs that cannot be improved upon by a coalition of the participants.

It satisfies the following 2 properties:

- **Efficiency**:
  The payoffs are distributed such that it is not possible to make any participant better off
  without making another one worse off.
  $$\displaystyle\sum_{x_i\in D} v_u(x_i) = v_u(D)\,$$

- **Coalitional rationality**:
  The sum of payoffs to the agents in any coalition S is at
  least as large as the amount that these agents could earn by
  forming a coalition on their own.
  $$\displaystyle\sum_{x_i\in S} v_u(x_i) \geq v_u(S), \forall S \subseteq D\,$$

The second property states that the sum of payoffs to the agents in any subcoalition S is at
least as large as the amount that these agents could earn by
forming a coalition on their own.

Least Core values
^^^^^^^^^^^^^^^^^

Unfortunately, for many cooperative games the Core may be empty.
By relaxing the coalitional rationality property by $e \gt 0$,
we are then able to find approximate payoffs:

$$\displaystyle\sum_{x_i\in S} v_u(x_i) + e \geq v_u(S), \forall S \subseteq D\,$$

The least core value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
by solving the following Linear Program:

$$
\begin{array}{lll}
\text{minimize} & \displaystyle{e} & \\
\text{subject to} & \displaystyle\sum_{x_i\in D} v_u(x_i) = v_u(D) & \\
& \displaystyle\sum_{x_i\in S} v_u(x_i) + e \geq v_u(S) &, \forall S \subseteq D \\
\end{array}
$$

Exact Least Core
----------------

This first algorithm is just a verbatim implementation of the definition. As such
it returns as exact a value as the utility function allows (see what this means
in :ref:`problems of data values`).

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.least_core import exact_least_core
   model = ...
   dataset = Dataset(...)
   utility = Utility(data, model)
   values = exact_least_core(utility)

Monte Carlo Least Core
----------------------

Because the number of subsets $S \subseteq D \setminus \{x_i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations.

The simplest approximation consists of two relaxations of the Least Core (:footcite:t:`yan_procaccia_2021`):

- Further relaxing the coalitional rationality property by a constant value $\epsilon > 0$:

  $$
  \sum_{x_i\in S} v_u(x_i) + e + \epsilon \geq v_u(S)
  $$

- Using a fraction of all subsets instead of all possible subsets.

Combined, this gives us the following property:

$$
P_{S\sim D}\left[\sum_{x_i\in S} v_u(x_i) + e^{*} + \epsilon \geq v_u(S)\right] \geq 1 - \delta
$$

Where $e^{*}$ is the optimal least core value.

With these relaxations, we obtain a polynomial running time.

Using Owen sampling follows the same pattern as every other method for Shapley
values in pyDVL. First construct the dataset and utility, then call
:func:`~pydvl.value.shapley.compute_shapley_values`:

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.least_core import montecarlo_least_core
   model = ...
   dataset = Dataset(...)
   max_iterations = ...
   assert max_iterations >= len(dataset)
   utility = Utility(data, model)
   values = montecarlo_least_core(utility, max_iterations=max_iterations)

.. note::

   ``max_iterations`` needs to be at least equal to the number of data points.

Other methods
=============

There are other game-theoretic concepts in pyDVL's roadmap, based on the notion
of semivalue, which is a generalization to different weighting schemes: in particular
**Banzhaf indices** and **Beta Shapley**, with better numerical and rank stability in
certain situations.

Contributions are welcome!


.. _problems of data values:

Problems of data values
=======================

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

  pyDVL offers a dedicated :func:`function composition
  <pydvl.utils.types.compose_score>` for scorer functions which can be used to
  squash a score. The following is defined in module :mod:`~pydvl.utils.numeric`:

  .. code-block:: python

     def sigmoid(x: float) -> float:
         return float(1 / (1 + np.exp(-x)))

     squashed_r2 = compose_score("r2", sigmoid, "squashed r2")

     squashed_variance = compose_score(
         "explained_variance", sigmoid, "squashed explained variance"
     )

  These squashed scores can prove useful in regression problems, but they can
  also introduce issues in the low-value regime.

* **High variance utility**: Classical applications of game theoretic value
  concepts operate with deterministic utilities, but in ML we use an evaluation
  of the model on a validation set as a proxy for the true risk. Even if the
  utility *is* bounded, if it has high variance then values will also have high
  variance, as will their Monte Carlo estimates. One workaround in pyDVL is to
  configure the caching system to allow multiple evaluations of the utility for
  every index set. A moving average is computed and returned once the standard
  error is small, see :class:`~pydvl.utils.config.MemcachedConfig`.

  :footcite:t:`wang_data_2022` prove that by relaxing one of the Shapley axioms
  and considering the general class of semi-values, of which Shapley is an
  instance, one can prove that a choice of constant weights is the best one can
  do in a utility-agnostic setting. So-called *Data Banzhaf* is on our to-do
  list!

* **Data set size**: Computing exact Shapley values is NP-hard, and Monte Carlo
  approximations can converge slowly. Massive datasets are thus impractical, at
  least with current techniques. A workaround is to group samples and investigate
  their value together. In pyDVL you can do this using
  :class:`~pydvl.utils.dataset.GroupedDataset`. There is a fully worked-out
  :doc:`example here <examples/shapley_basic_spotify>`. Some algorithms also
  provide different sampling strategies to reduce the variance, but due to a
  no-free-lunch-type theorem, no single strategy can be optimal for all
  utilities.

* **Model size**: Since every evaluation of the utility entails retraining the
  whole model on a subset of the data, large models require great amounts of
  computation. But also, they will effortlessly interpolate small to medium
  datasets, leading to great variance in the evaluation of performance on the
  dedicated validation set. One mitigation for this problem is cross-validation,
  but this would incur massive computational cost. As of v.0.3.0 there are no
  facilities in pyDVL for cross-validating the utility (note that this would
  require cross-validating the whole value computation).

References
==========

.. footbibliography::
