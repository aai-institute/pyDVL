.. _data values:

=====================
Computing data values
=====================

Data value is a 


For data valuation, one uses the functions in modules
:mod:`~pydvl.shapley` and :mod:`~pydvl.loo`, supported by
:mod:`pydvl.utils.dataset` and :mod:`~pydvl.utils.utility`, as detailed below.

Creating a Dataset
==================

The class :class:`~pydvl.utils.dataset.Dataset` is a simple convenience wrapper
for the train and test splits that is used throughout pyDVL. It can be used as
follows:

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

It is also possible to construct Datasets from sklearn datasets for illustrative
purposes using :meth:`~pydvl.utils.dataset.Dataset.from_sklearn`.

Creating a Utility
==================

The :class:`~pydvl.utils.utility.Utility` class is a convenient wrapper for the
dataset, model and scoring function which is used for valuation methods like
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

The Leave-One-Out method is a naive approach that should only be used for testing purposes.

.. code-block:: python

   >>> from pydvl.loo.naive import naive_loo
   >>> utility = Utility(...)
   >>> values = naive_loo(utility)


Computing Shapley values
========================

The Shapley method is a game-theoretic approach to compute data values.
Here we use Truncated Montecarlo Shapley because it is the most efficient.

.. code-block:: python

   >>> from pydvl.utils import Utility
   >>> from pydvl.shapley.montecarlo import truncated_montecarlo_shapley
   >>> from pydvl.reporting.plots import shapley_results
   >>> utility = Utility(...)
   >>> values, errors = truncated_montecarlo_shapley(u=utility, max_iterations=100)
   >>> scores = compute_fb_scores(model=utility.model, data=utility.data, values=values)
   >>> shapley_results(scores)


