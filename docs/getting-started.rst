Getting started
===============

In order to use the library, you need to use `Memcached <https://memcached.org/>`_,
an in-memory key-value store for small chunks of arbitrary data (strings, objects),
in order to cache certain results and speed-up the computations.

You can either install it directly on your system and run it. For that refer to the
`Getting Started section <https://github.com/memcached/memcached/wiki#getting-started>`_
of Memcached's wiki.

Or you can run it inside a container:

.. code-block:: shell

    docker container run -it --rm -p 11211:11211 memcached:latest -v

Caching is enabled by default but can be disabled if not needed or desired.

Creating Dataset
----------------

.. code-block:: python

   >>> import numpy as np
   >>> from valuation.utils import Dataset
   >>> from sklearn.model_selection import train_test_split
   >>> X, y = np.arange(100).reshape((50, 2)), np.arange(50)
   >>> X_train, X_test, y_train, y_test = train_test_split(
   ...     X, y, test_size=0.5, random_state=16
   ... )
   ...
   >>> dataset = Dataset(X_train, X_test, y_train, y_test)


Creating Utility
----------------

.. code-block:: python

   >>> from valuation.utils import Dataset, Utility
   >>> from sklearn.linear_model import LinearRegression
   >>> dataset = Dataset(...)
   >>> model = LinearRegression()
   >>> utility = Utility(model, dataset)


Computing Shapley values
------------------------

.. code-block:: python

   >>> from valuation.utils import map_reduce, Utility
   >>> from valuation.shapley.montecarlo import combinatorial_montecarlo_shapley
   >>> from valuation.reporting.scores import compute_fb_scores
   >>> from valuation.reporting.plots import shapley_results
   >>> utility = Utility(...)
   >>> fun = partial(truncated_montecarlo_shapley, utility=utility, progress=True)
   >>> values_nmcs, hist_nmcs = map_reduce(fun, num_runs=10, num_jobs=160)
   >>> scores_nmcs = compute_fb_scores(model=model, data=data, values=values_nmcs)
   >>> shapley_results(scores_nmcs)
