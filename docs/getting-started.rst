.. _getting started:

===============
Getting started
===============

Make sure you have :ref:`installed pyDVL <pyDVL Installation>` before proceeding further.

In order to use the library, you need to use `Memcached <https://memcached.org/>`_,
an in-memory key-value store for small chunks of arbitrary data (strings, objects),
in order to cache certain results and speed-up the computations.

You can either install it directly on your system and run it. For that refer to the
`Getting Started section <https://github.com/memcached/memcached/wiki#getting-started>`_
of Memcached's wiki. You can run it using:

.. code-block:: shell

   $ memcached -u user

Or you can run it inside a container:

.. code-block:: shell

    $ docker container run -it --rm -p 11211:11211 memcached:latest -v

Caching is enabled by default but can be disabled if not needed or desired.

Creating Dataset
================

The :class:`~valuation.utils.dataset.Dataset` class is a convenient wrapper
for the train and test splits that is used throughout the codebase. It can be used as follows:

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
================

The :class:`~valuation.utils.utility.Utility` class is a convenient wrapper
for the dataset, model and scoring function. It is used in the Leave-One-Out and Shapley methods.
It can be used as follows:

.. code-block:: python

   >>> from valuation.utils import Dataset, Utility
   >>> from sklearn.linear_model import LinearRegression
   >>> dataset = Dataset(...)
   >>> model = LinearRegression()
   >>> utility = Utility(model, dataset)


Computing Leave-One-Out values
==============================

The Leave-One-Out method is a naive approach that should only be used for testing purposes.

.. code-block:: python

   >>> from valuation.loo.naive import naive_loo
   >>> utility = Utility(...)
   >>> values = naive_loo(utility)


Computing Shapley values
========================

The Shapley method is a game-theoretic approach to compute data valuation.
Here we use Truncated Montecarlo Shapley because it is the most efficient.

.. code-block:: python

   >>> from valuation.utils import Utility
   >>> from valuation.shapley.montecarlo import truncated_montecarlo_shapley
   >>> from valuation.reporting.plots import shapley_results
   >>> utility = Utility(...)
   >>> values, errors = truncated_montecarlo_shapley(u=utility, max_iterations=100)
   >>> scores = compute_fb_scores(model=utility.model, data=utility.data, values=values)
   >>> shapley_results(scores)


Computing Influence values
==========================

There are two possibilities to calculate influences. For linear regression the influences can be calculated via the
direct analytical function (this is used in testing as well). For more general models or loss functions
one can use the ``TwiceDifferentiable`` protocol, which provides the required methods for calculating the influences.
In general there are two types of influences, namely Up-weighting and Perturbation influences. Each method supports
the choice of one ot them by pinning an enumeration in the parameters. Furthermore, we distinguish between the following types of calculations.

Direct linear influences
------------------------

These can only applied to a regression problem where x and y are from the real numbers. When
a Dataset object is available, this is as simple as calling

.. code-block:: python

   >>> from valuation.influence.linear import linear_influences
   >>> linear_influences(dataset)


the linear influence functions. Internally these method fit a linear regression model and use this
to subsequently calculate the influences. Take a closer look at their inner definition, to reuse a model
in calculation or optimize the calculation for your specific application.

Exact influences using TwiceDifferentiable protocol
---------------------------------------------------

If you create a model, which supports the ``TwiceDifferentiable`` protocol. This means that it is
capable of calculating second derivative matrix vector products and gradients with respect to the
loss and data samples.

.. code-block:: python

   >>> from valuation.influence.general import influences
   >>>
   >>> influences(
   ...    model,
   ...    dataset,
   ... )


Influences using TwiceDifferentiable protocol and approximate matrix inversion
------------------------------------------------------------------------------

Sometimes it is not possible to construct the complete Hessian in RAM.
In that case one can use conjugate gradient as a space-efficient
approximation to inverting the full matrix. In pyDVL this can be done
by adding ``inversion_method`` parameter to the influences function call.


.. code-block:: python

   >>> from valuation.influence.general import influences

   >>> influences(
   ...     model,
   ...     dataset,
   ...     inversion_method="cg"
   ... )


Perturbation influences
-----------------------

All previous mentioned influences can be calculated feature-wise by adding ``influence_type`` parameter
to the influences function call.

.. code-block:: python

   >>> from valuation.influence.general import influences
   >>>
   >>> influences(
   ...     model,
   ...     dataset,
   ...     influence_type='perturbation'
   ... )

What's next
===========

You should go to the :ref:`Examples <examples>` section of the documentation
to see more detailed usage of the library.
