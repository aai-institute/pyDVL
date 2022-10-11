.. _influence:

==========================
Computing influence values
==========================

There are two ways to compute influences. For linear regressions, the influences can be computed
analytically (this is used in testing as well). For more general models or loss functions,
we can use the ``TwiceDifferentiable`` protocol, which provides the required methods for computing the influences.

In general there are two types of influences, namely Up-weighting and Perturbation influences.
Each method supports the choice of one ot them by pinning an enumeration in the parameters.
Furthermore, we distinguish between the following types of calculations:

Direct linear influences
------------------------

These can only be applied to a regression problem where x and y are real numbers.

.. code-block:: python

   >>> from pydvl.influence.linear import compute_linear_influences
   >>> compute_linear_influences(
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test
   ... )


Internally this method fits a linear regression model and uses it
to subsequently calculate the influences. Take a closer look at their inner definition, to reuse a model
in calculation or optimize the calculation for your specific application.

Exact influences using TwiceDifferentiable protocol
---------------------------------------------------

If you create a model, which supports the ``TwiceDifferentiable`` protocol. This means that it is
capable of calculating second derivative matrix vector products and gradients with respect to the
loss and data samples.

.. code-block:: python

   >>> from pydvl.influence import influences
   >>> compute_influences(
   ...    model,
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test,,
   ... )


Influences using TwiceDifferentiable protocol and approximate matrix inversion
------------------------------------------------------------------------------

Sometimes it is not possible to construct the complete Hessian in memory.
In that case one can use conjugate gradient as a space-efficient
approximation to inverting the full matrix. In pyDVL this can be done
by adding ``inversion_method`` parameter to the influences function call.


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

All previous mentioned influences can be calculated feature-wise by adding ``influence_type`` parameter
to the influences function call.

.. code-block:: python

   >>> from pydvl.influence import compute_influences
   >>>
   >>> compute_influences(
   ...    model,
   ...    x_train,
   ...    y_train,
   ...    x_test,
   ...    y_test,
   ...    influence_type="perturbation"
   ... )
