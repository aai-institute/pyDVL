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
the modules :mod:`~pydvl.shapley` and :mod:`~pydvl.loo`, supported by
:mod:`pydvl.utils.dataset` and :mod:`~pydvl.utils.utility`, as detailed below.

Creating a Dataset
==================

The first item in the tuple $(D, \mathcal{A}, u)$ characterising data value is
the dataset. The class :class:`~pydvl.utils.dataset.Dataset` is a simple
convenience wrapper for the train and test splits that is used throughout pyDVL.

It can be used as follows:

.. code-block:: python

   >>> import numpy as np
   >>> from pydvl.utils import Dataset
   >>> from sklearn.model_selection import train_test_split
   >>> X, y = np.arange(100).reshape((50, 2)), np.arange(50)
   >>> X_train, X_test, y_train, y_test = train_test_split(
   ...     X, y, test_size=0.5, random_state=16
   ... )
   ...
   >>> dataset = Dataset(X_train, X_test, y_train, y_test)

It is also possible to construct Datasets from sklearn toy datasets for
illustrative purposes using :meth:`~pydvl.utils.dataset.Dataset.from_sklearn`.

Creating a Utility
==================

In order to keep track of all three items in $(D, \mathcal{A}, u)$ we use the
class :class:`~pydvl.utils.utility.Utility`. This is a convenient wrapper for
the dataset, model and scoring function which is used for valuation methods like
:mod:`Leave-One-Out<pydvl.loo>` and :mod:`Shapley<pydvl.shapley>`.

It can be used as follows:

.. code-block:: python

   >>> from pydvl.utils import Dataset, Utility
   >>> from sklearn.linear_model import LinearRegression
   >>> dataset = Dataset(...)
   >>> model = LinearRegression()
   >>> utility = Utility(model, dataset)


Computing Leave-One-Out values
==============================

The Leave-One-Out method is a naive approach that should only be used for
testing purposes. One particular weakness is that it does not necessarily
correlate with an intrinsic value of a sample: since it is only marginal utility,
it can happen that the training set is large enough for a single sample not to
have any significant effect on training performance, despite any qualities it
may possess. Whether this is indicative of low value or not depends on each
one's goals and definitions.

.. code-block:: python

   >>> from pydvl.loo.naive import naive_loo
   >>> utility = Utility(...)
   >>> values = naive_loo(utility)


Computing Shapley values
========================

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
one is done via Monte Carlo sampling of the powerset $\mathcal{P}(D)$.

However, an equivalent formulation of the expression above is typically used
which uses permutations over indices instead of subsets:

$$v_u(x_i) = \frac{1}{n!} \sum_{\sigma \in \Pi(n)} [u(\sigma_{i-1} \cup {i}) − u(\sigma_{i})],$$

where $\sigma_i$ denotes the set of indices in permutation sigma up until the
position of index $i$. There exist variations and different sampling strategies
for both formulations in the literature.

Then one does Monte Carlo sampling of permutations. By adding early
stopping, the result is the so-called *Truncated Monte Carlo Shapley*, which is
efficient and has proven useful in some applications:

.. code-block:: python

   >>> from pydvl.utils import Utility
   >>> from pydvl.shapley import compute_shapley_values
   >>> utility = Utility(...)
   >>> df = compute_shapley_values(
           u=utility, mode="truncated_montecarlo", max_iterations=100
       )

The code above will generate a
`pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
with values and estimated standard errors. Please refer to the documentation in
:mod:`pydvl.shapley` for more information.

Other methods
=============

Other game-theoretic concepts in pyDVL's roadmap are the **Least Core**, and
**Banzhaf indices** (the latter is just a different weighting scheme with better
numerical stability properties). Contributions are welcome!
