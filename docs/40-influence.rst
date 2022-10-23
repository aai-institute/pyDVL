.. _influence:

==========================
Computing influence values
==========================

.. todo::

   This section needs rewriting:
    - Introduce some theory
    - Explain how the methods differ
    - Add example for `TwiceDifferentiable`
    - Improve uninformative examples

There are two ways to compute influences. For linear regression, the influences
can be computed analytically. For more general models or loss functions, one can
implement the :class:`TwiceDifferentiable` protocol, which provides the required
methods for computing the influences.

pyDVL supports two ways of computing the empirical influence function, namely
up-weighting of samples and perturbation influences. The choice is done by a
parameter in the call to the main entry points,
:func:`~pydvl.influence.linear.compute_linear_influences` and
:func:`~pydvl.influence.compute_influences`.

Influence for OLS
-----------------
.. warning::

   This will be deprecated. It makes no sense to have a separate interface for
   linear models.

Because the Hessian of the least squares loss for a regression problem can be
computed analytically, we provide
:func:`~pydvl.influence.linear.compute_linear_influences` as a convenience
function to work with these models.

.. code-block:: python

   >>> from pydvl.influence.linear import compute_linear_influences
   >>> compute_linear_influences(
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test
   ... )


This method calculates the influence function for each sample in x_train for a
least squares regression problem.


Exact influences using the `TwiceDifferentiable` protocol
---------------------------------------------------------

More generally, influences can be computed for any model which implements the
:class:`TwiceDifferentiable` protocol, i.e. which is capable of calculating
second derivative matrix vector products and gradients of the loss evaluated on
training and test samples.

.. code-block:: python

   >>> from pydvl.influence import influences
   >>> compute_influences(
   ...    model,
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test,,
   ... )


Approximate matrix inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is not possible to construct the complete Hessian in memory. In
that case one can use conjugate gradient as a space-efficient approximation to
inverting the full matrix. In pyDVL this can be done with the parameter
`inversion_method` of :func:`~pydvl.influence.compute_influences`:


.. code-block:: python

   >>> from pydvl.influence import compute_influences

   >>> compute_influences(
   ...    model,
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test,
   ...    inversion_method="cg"
   ... )


Perturbation influences
-----------------------

As mentioned, the method of empirical influence computation can be selected
in :func:`~pydvl.influence.compute_influences` with `influence_type`:

.. code-block:: python

   >>> from pydvl.influence import compute_influences
   >>> compute_influences(
   ...    model,
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test,
   ...    influence_type="perturbation"
   ... )
