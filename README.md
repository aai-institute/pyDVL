# pyDVL

[![CI](https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml/badge.svg)](https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml) 


Welcome to the pyDVL library for data valuation!

Refer to our [documentation](https://appliedAI-Initiative.github.io/valuation) for more detailed information.

# Installation

To install the latest release use:

```shell
$ pip install pydvl
```

For more instructions and information refer to the [Installing pyDVL section](https://appliedAI-Initiative.github.io/valuation/install.html)
of the documentation.

# Usage

pyDVL requires Memcached in order to cache certain results and speed-up computation.

You need to run it either locally or using Docker:

```shell
docker container run -it --rm -p 11211:11211 memcached:latest -v
```

Caching is enabled by default but can be disabled if not needed or desired. 

For more instructions and information refer to the [Getting Started section](https://appliedAI-Initiative.github.io/valuation/getting-started.html) 
of the documentation 

Refer to the notebooks in the [notebooks](notebooks) folder for usage examples.

## Influence functions

To use all features of influence functions execute ```pip install pyDVL[influence]```. It is noteworthy to say that
this includes heavy autograd frameworks and thus is left out by default. There are two possibilities to 
calculate influences. For linear regression the influences can be calculated via the
direct analytical function (this is used in testing as well). For more general models or loss functions
one can use the ```TwiceDifferentiable``` protocol, which provides the required methods for calculating the influences.
In general there are two types of influences, namely Up-weighting and Perturbation influences. Each method supports 
the choice of one ot them by pinning an enumeration in the parameters. Furthermore, we distinguish between the following types of calculations.

### Direct linear influences

These can only applied to a regression problem where x and y are from the real numbers. When
a Dataset object is available, this is as simple as calling

```python

from valuation.influence.linear import linear_influences
from valuation.influence.types import InfluenceTypes

linear_influences(
    x_train,
    y_train,
    x_test,
    y_test, 
    influence_type=InfluenceTypes.Up
)
```

the linear influence functions. Internally these method fit a linear regression model and use this
to subsequently calculate the influences. Take a closer look at their inner definition, to reuse a model
in calculation or optimize the calculation for your specific application.

### Exact influences using TwiceDifferentiable protocol

If you create a model, which supports the ```TwiceDifferentiable``` protocol. This means that it is 
capable of calculating second derivative matrix vector products and gradients with respect to the
loss and data samples.

```python

from valuation.influence.general import influences
from valuation.influence.types import InfluenceTypes

influences(
   model,
   x_train,
   y_train,
   x_test,
   y_test, 
   influence_type=InfluenceTypes.Up
)
```

### Influences using TwiceDifferentiable protocol and approximate matrix inversion

Sometimes it is not possible to construct the complete Hessian in RAM.
In that case one can use conjugate gradient as a space-efficient
approximation to inverting the full matrix. In pyDVL this can be done
by adding a parameter to the influences function call.

```python
from valuation.influence.general import influences
from valuation.influence.types import InfluenceTypes

influences(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    influence_type=InfluenceTypes.Up,
    use_conjugate_gradient=True
)
```

# Contributing

Please open new issues for bugs, feature requests and extensions. See more details about the structure and
workflow in the [developer's readme](README-dev.md).

# To do

* fix all 'em broken things.
* pytest plugin for algorithms with epsilon,delta guarantees:
  run n times, expect roughly n*delta failures at most.
