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
the modules :mod:`~pydvl.value.shapley` and :mod:`~pydvl.value.loo`, supported
by :mod:`pydvl.utils.dataset` and :mod:`~pydvl.utils.utility`, as detailed below.

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

It can be often useful or necessary to group samples so as to valuate them
together. Be it because data valuation methods are computationally very
expensive, or because we are interested in the groups themselves,
:class:`~pydvl.utils.dataset.GroupedDataset` provides an alternative to
`Dataset` with the same interface which allows this.

You can see an example in action in the
:doc:`Spotify notebook <examples/shapley_basic_spotify>`.

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
greatly reduces computation times of Monte Carlo methods using Permutations.
You can read more about :ref:`caching setup` in the installation guide.

Learning the utility
^^^^^^^^^^^^^^^^^^^^

Because each evaluation of the utility entails a full retrain of the model with
a new subset of the training set, it is natural to try to learn this mapping
from subsets to scores. This is the idea behind **Data Utility Learning (DUL)**
[1]_ and in pyDVL it's as simple as wrapping the `Utility` inside
:class:`~pydvl.utils.utility.DataUtilityLearning`:

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

The Leave-One-Out method is a naive approach that should only be used for
testing purposes. One particular weakness is that it does not necessarily
correlate with an intrinsic value of a sample: since it is only marginal utility,
it can happen that the training set is large enough for a single sample not to
have any significant effect on training performance, despite any qualities it
may possess. Whether this is indicative of low value or not depends on each
one's goals and definitions.

.. code-block:: python

   from pydvl.value.loo.naive import naive_loo
   utility = Utility(...)
   values = naive_loo(utility)


.. _Shapley:

Shapley values
==============

The Shapley method is an approach to compute data values originating in
cooperative game theory. Shapley values are a common way of assigning payoffs to
each participant in a cooperative game (i.e. one in which players can form
coalitions) in a way that ensures that certain axioms are fulfilled.

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is computed
as a weighted sum of its marginal utility wrt. every possible coalition of
training samples within the training set:

$$v_u(x_i) = \frac{1}{n} \sum_{S \subseteq D \setminus \{x_i\}} \binom{n-1}{ | S | }^{-1} [u(S \cup \{x_i\}) − u(S)] ,$$

Because the number of subsets $S \subseteq D \setminus \{x_i\}$ is
$2^{ | D | - 1 }$, one typically must resort to approximations. The simplest
one is done via Monte Carlo sampling of the powerset $\mathcal{P}(D)$. In pyDVL
this simple technique is called "Combinatorial Monte Carlo" and can be accessed,
together with all others, via a common interface provided by
:func:`~pydvl.value.shapley.compute_shapley_values`. However, the method is very
slow to converge and others are preferred.

An algorithm which can be used in practice is **Owen Sampling** [2]_. It
introduces a continuous extension of the utility from $\{0,1\}^n$ to $[0,1]^n$.
The ensuing expression for Shapley value uses integration instead of discrete
weights:

$$v_u(i) = \int_0^1 \mathbb{E}_{S \sim P_q(D_{\backslash \{ i \}})} [u(S \cup {i}) - u(S)].$$

Using Owen sampling follows the same pattern as every other method for Shapley
values in pyDVL. First construct the utility

.. code-block:: python

   from pydvl.utils import Dataset, Utility
   from pydvl.value.shapley import compute_shapley_values
   dataset = Dataset(...)
   model = ...
   utility = Utility(data, model)
   df = compute_shapley_values(
           u=utility, mode="owen_sampling", max_iterations=100
       )

The code above will generate a
`pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
with values and estimated standard errors. Please refer to the documentation in
:mod:`pydvl.value.shapley` for more information. There are more details on Owen
sampling, and its variant *Halved Owen Sampling* in the documentation for the
function doing the work behind the scenes:
:func:`~pydvl.value.shapley.montecarlo.owen_sampling_shapley`.

An equivalent way of computing Shapley values appears often in the literature.
It uses permutations over indices instead of subsets:

$$v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)} [u(\sigma_{i-1} \cup {i}) − u(\sigma_{i})],$$

where $\sigma_i$ denotes the set of indices in permutation sigma up until the
position of index $i$. To approximate this sum (with $\mathcal{O}(n!)$ terms!)
one uses Monte Carlo sampling of permutations, something which has suprisingly
low sample complexity. By adding early stopping, the result is the so-called
**Truncated Monte Carlo Shapley** [3]_, which is efficient and has proven useful
in some applications.

Usage follows the same pattern as above:

.. code-block:: python

   from pydvl.utils import Utility
   from pydvl.value.shapley import compute_shapley_values
   utility = Utility(...)
   df = compute_shapley_values(
           u=utility, mode="truncated_montecarlo", max_iterations=100
       )


Other methods
=============

Other game-theoretic concepts in pyDVL's roadmap are the **Least Core**, and
**Banzhaf indices** (the latter is just a different weighting scheme with better
numerical stability properties). Contributions are welcome!

References
==========

.. [1] Wang, Tianhao, Yu Yang, and Ruoxi Jia. ‘Improving Cooperative Game
   Theory-Based Data Valuation via Data Utility Learning’. arXiv, 2022.
   https://doi.org/10.48550/arXiv.2107.06336.
.. [2] Okhrati, Ramin, and Aldo Lipani. ‘A Multilinear Sampling Algorithm
   to Estimate Shapley Values’. In 2020 25th International Conference on
   Pattern Recognition (ICPR), 7992–99. IEEE, 2021.
   https://doi.org/10.1109/ICPR48806.2021.9412511.
.. [3] Ghorbani, Amirata, and James Zou. ‘Data Shapley: Equitable Valuation of
   Data for Machine Learning’. In International Conference on Machine Learning,
   2242–51. PMLR, 2019. http://proceedings.mlr.press/v97/ghorbani19c.html.
