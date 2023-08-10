---
title: Computing Influence Values
alias: 
  name: influence-values
  text: Computing Influence Values
---

# Computing influence values

!!! Warning 
    Much of the code in the package [pydvl.influence][pydvl.influence]
    is experimental or untested. Package structure and basic API are bound
    to change before v1.0.0

The influence function (IF) is a method to quantify the effect (influence) that
each training point has on the parameters of a model, and by extension on any
function thereof. In particular, it allows to estimate how much each training
sample affects the error on a test point, making the IF useful for understanding
and debugging models.

pyDVL implements several methods for the efficient computation of the IF for
machine learning.

# The Influence Function

First introduced in the context of robust statistics in
[@hampel1974influence], the IF was popularized in the context of
machine learning in [@koh_understanding_2017]. 

Following their formulation, consider an input space $\mathcal{X}$
(e.g. images) and an output space $\mathcal{Y}$ (e.g. labels). 
Let's take $z_i = (x_i, y_i)$, for $i \in  \{1,...,n\}$ to be
the $i$-th training point, and $\theta$ to be the
(potentially highly) multi-dimensional parameters of a model
(e.g. $\theta$ is a big array with all of a neural network's parameters,
including biases and/or dropout rates).
We will denote with $L(z, \theta)$ the loss of the model for
point $z$ when the parameters are $\theta.$

To train a model, we typically minimize the loss over all $z_i$, i.e. the
optimal parameters are

$$\hat{\theta} = \arg \min_\theta \sum_{i=1}^n L(z_i, \theta).$$

In practice, lack of convexity means that one doesn't really obtain the
minimizer of the loss, and the training is stopped when the validation loss
stops decreasing.

For notational convenience, let's define

$$\hat{\theta}_{-z} = \arg \min_\theta \sum_{z_i \ne z} L(z_i, \theta), $$

i.e. $\hat{\theta}_{-z}$ are the model parameters that minimize the total loss
when $z$ is not in the training dataset.

In order to compute the impact of each training point on the model, we would
need to calculate $\hat{\theta}_{-z}$ for each $z$ in the training dataset, thus
re-training the model at least ~$n$ times (more if model training is
stochastic). This is computationally very expensive, especially for big neural
networks. To circumvent this problem, we can just calculate a first order
approximation of $\hat{\theta}$. This can be done through single backpropagation
and without re-training the full model.



There are two ways to compute influences. For linear regression, the influences
can be computed analytically. For more general models or loss functions, one can
implement the [TwiceDifferentiable][pydvl.influence.frameworks.torch_differentiable.TwiceDifferentiable]
protocol, which provides the required  methods for computing the influences.

pyDVL supports two ways of computing the empirical influence function, namely
up-weighting of samples and perturbation influences. The choice is done by a
parameter in the call to the main entry points,
[compute_linear_influences][pydvl.influence.linear.compute_linear_influences]
and [compute_influences][pydvl.influence.compute_influences].

## Approximating the influence of a point

Let's define

$$\hat{\theta}_{\epsilon, z} = \arg \min_\theta \frac{1}{n}\sum_{i=1}^n L(z_i,
\theta) + \epsilon L(z, \theta), $$

which is the optimal $\hat{\theta}$ when we up-weight $z$ by an amount $\epsilon
\gt 0$.

From a classical result (a simple derivation is available in Appendix A of
[@koh_understanding_2017]), we know that:

$$\frac{d \ \hat{\theta}_{\epsilon, z}}{d \epsilon} \Big|_{\epsilon=0} =
-H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}), $$

where $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^n \nabla_\theta^2 L(z_i,
\hat{\theta})$ is the Hessian of $L$. These quantities are also knows as
**influence factors**.

Importantly, notice that this expression is only valid when $\hat{\theta}$ is a
minimum of $L$, or otherwise $H_{\hat{\theta}}$ cannot be inverted! At the same
time, in machine learning full convergence is rarely achieved, so direct Hessian
inversion is not possible. Approximations need to be developed that circumvent
the problem of inverting the Hessian of the model in all those (frequent) cases
where it is not positive definite.

The influence of training point $z$ on test point $z_{\text{test}}$ is defined
as:

$$\mathcal{I}(z, z_{\text{test}}) =  L(z_{\text{test}}, \hat{\theta}_{-z}) -
L(z_{\text{test}}, \hat{\theta}). $$

Notice that $\mathcal{I}$ is higher for points $z$ which positively impact the
model score, since the loss is higher when they are excluded from training. In
practice, one needs to rely on the following infinitesimal approximation:

$$\mathcal{I}_{up}(z, z_{\text{test}}) = - \frac{d L(z_{\text{test}},
\hat{\theta}_{\epsilon, z})}{d \epsilon} \Big|_{\epsilon=0} $$

Using the chain rule and the results calculated above, we get:

$$\mathcal{I}_{up}(z, z_{\text{test}}) = - \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ \frac{d \hat{\theta}_{\epsilon, z}}{d \epsilon}
\Big|_{\epsilon=0} = \nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top \
H_{\hat{\theta}}^{-1} \ \nabla_\theta L(z, \hat{\theta}) $$

All the resulting factors are gradients of the loss wrt. the model parameters
$\hat{\theta}$. This can be easily computed through one or more backpropagation
passes.

## Perturbation definition of the influence score

How would the loss of the model change if, instead of up-weighting an individual
point $z$, we were to up-weight only a single feature of that point? Given $z =
(x, y)$, we can define $z_{\delta} = (x+\delta, y)$, where $\delta$ is a vector
of zeros except for a 1 in the position of the feature we want to up-weight. In
order to approximate the effect of modifying a single feature of a single point
on the model score we can define

$$\hat{\theta}_{\epsilon, z_{\delta} ,-z} = \arg \min_\theta
\frac{1}{n}\sum_{i=1}^n L(z_{i}, \theta) + \epsilon L(z_{\delta}, \theta) -
\epsilon L(z, \theta), $$

Similarly to what was done above, we up-weight point $z_{\delta}$, but then we
also remove the up-weighting for all the features that are not modified by
$\delta$. From the calculations in
[the previous section](#approximating-the-influence-of-a-point),
it is then easy to see that

$$\frac{d \ \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d \epsilon}
\Big|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \nabla_\theta \Big( L(z_{\delta},
\hat{\theta}) - L(z, \hat{\theta}) \Big) $$

and if the feature space is continuous and as $\delta \to 0$ we can write

$$\frac{d \ \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d \epsilon}
\Big|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \ \nabla_x \nabla_\theta L(z,
\hat{\theta}) \delta + \mathcal{o}(\delta) $$

The influence of each feature of $z$ on the loss of the model can therefore be
estimated through the following quantity:

$$\mathcal{I}_{pert}(z, z_{\text{test}}) = - \lim_{\delta \to 0} \
\frac{1}{\delta} \frac{d L(z_{\text{test}}, \hat{\theta}_{\epsilon, \
z_{\delta}, \ -z})}{d \epsilon} \Big|_{\epsilon=0} $$

which, using the chain rule and the results calculated above, is equal to

$$\mathcal{I}_{pert}(z, z_{\text{test}}) = - \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ \frac{d \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d
\epsilon} \Big|_{\epsilon=0} = \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ H_{\hat{\theta}}^{-1} \ \nabla_x \nabla_\theta L(z,
\hat{\theta}) $$

The perturbation definition of the influence score is not straightforward to
understand, but it has a simple interpretation: it tells how much the loss of
the model changes when a certain feature of point z is up-weighted. A positive
perturbation influence score indicates that the feature might have a positive
effect on the accuracy of the model.

It is worth noting that the perturbation influence score is a very rough
estimate of the impact of a point on the models loss and it is subject to large
approximation errors. It can nonetheless be used to build training-set attacks,
as done in [@koh_understanding_2017].

# Computing influences

The main entry point of the library for influence calculation is
[compute_influences][pydvl.influence.general.compute_influences].
Given a pre-trained pytorch model with a loss, first an instance of
[TorchTwiceDifferentiable][pydvl.influence.general.TorchTwiceDifferentiable]
needs to be created:

```python
from pydvl.influence import TorchTwiceDifferentiable
wrapped_model = TorchTwiceDifferentiable(model, loss, device)
```

The device specifies where influence calculation will be run. 

Given training and test data loaders, the influence of each training point on
each test point can be calculated via:

```python
from pydvl.influence import compute_influences
from torch.utils.data import DataLoader
training_data_loader = DataLoader(...)
test_data_loader = DataLoader(...)
compute_influences(
   wrapped_model,
   training_data_loader,
   test_data_loader,
)
```

The result is a tensor with one row per test point and one column per training
point. Thus, each entry $(i, j)$ represents the influence of training point $j$
on test point $i$. A large positive influence indicates that training point $j$
tends to improve the performance of the model on test point $i$, and vice versa,
a large negative influence indicates that training point $j$ tends to worsen the
performance of the model on test point $i$.

## Perturbation influences

The method of empirical influence computation can be selected in
[compute_influences][pydvl.influence.general.compute_influences]
with the parameter `influence_type`:

```python
from pydvl.influence import compute_influences
compute_influences(
   wrapped_model,
   training_data_loader,
   test_data_loader,
   influence_type="perturbation",
)
```

The result is a tensor with at least three dimensions. The first two dimensions
are the same as in the case of `influence_type=up` case, i.e. one row per test
point and one column per training point. The remaining dimensions are the same
as the number of input features in the data. Therefore, each entry in the tensor
represents the influence of each feature of each training point on each test
point.

## Approximate matrix inversion

In almost every practical application it is not possible to construct, even less
invert the complete Hessian in memory. pyDVL offers several approximate
algorithms to invert it by setting the parameter `inversion_method` of
[compute_influences][pydvl.influence.general.compute_influences].

```python
from pydvl.influence import compute_influences
compute_influences(
   wrapped_model,
   training_data_loader,
   test_data_loader,
   inversion_method="cg"
)
```

Each inversion method has its own set of parameters that can be tuned to improve
the final result. These parameters can be passed directly to
[compute_influences][pydvl.influence.general.compute_influences]
as keyword arguments. For example, the following code sets
the maximum number of iterations for conjugate
gradient to $100$ and the minimum relative error to $0.01$:

```python
from pydvl.influence import compute_influences
compute_influences(
   wrapped_model,
   training_data_loader,
   test_data_loader,
   inversion_method="cg",
   hessian_regularization=1e-4,
   maxiter=100,
   rtol=0.01
)
```

## Hessian regularization

Additionally, and as discussed in [the introduction](#the-influence-function),
in machine learning training rarely converges to a
global minimum of the loss. Despite good apparent convergence, $\hat{\theta}$
might be located in a region with flat curvature or close to a saddle point. In
particular, the Hessian might have vanishing eigenvalues making its direct
inversion impossible. Certain methods, such as the
[Arnoldi method](#arnoldi-solver) are robust against these problems,
but most are not.

To circumvent this problem, many approximate methods can be implemented.
The simplest adds a small *hessian perturbation term*,
i.e. $H_{\hat{\theta}} + \lambda \mathbb{I}$,
with $\mathbb{I}$ being the identity matrix. This standard trick
ensures that the eigenvalues of $H_{\hat{\theta}}$ are bounded away from zero
and therefore the matrix is invertible. In order for this regularization not to
corrupt the outcome too much, the parameter $\lambda$ should be as small as
possible while still allowing a reliable inversion of $H_{\hat{\theta}} +
\lambda \mathbb{I}$.

```python
from pydvl.influence import compute_influences
compute_influences(
   wrapped_model,
   training_data_loader,
   test_data_loader,
   inversion_method="cg",
   hessian_regularization=1e-4
)
```

## Influence factors

The [compute_influences][pydvl.influence.general.compute_influences]
method offers a fast way to obtain the influence scores given a model
and a dataset. Nevertheless, it is often more convenient
to inspect and save some of the intermediate results of
influence calculation for later use.

The influence factors(refer to
[the previous section](#approximating-the-influence-of-a-point) for a definition)
are typically the most computationally demanding part of influence calculation.
They can be obtained via the
[compute_influence_factors][pydvl.influence.general.compute_influence_factors]
function, saved, and later used for influence calculation
on different subsets of the training dataset.

```python
from pydvl.influence import compute_influence_factors
influence_factors = compute_influence_factors(
   wrapped_model,
   training_data_loader,
   test_data_loader,
   inversion_method="cg"
)
```

The result is an object of type 
[InverseHvpResult][pydvl.influence.framework.InverseHvpResult],
which holds the calculated influence factors (`influence_factors.x`) and a
dictionary with the info on the inversion process (`influence_factors.info`).

# Methods for inverse HVP calculation

In order to calculate influence values, pydvl implements several methods for the
calculation of the inverse Hessian vector product (iHVP). More precisely, given
a model, training data and a tensor $b$, the function
[solve_hvp][pydvl.influence.inversion.solve_hvp]
will find $x$ such that $H x = b$,
with $H$ is the hessian of model.

Many different inversion methods can be selected via the parameter 
`inversion_method` of
[compute_influences][pydvl.influence.general.compute_influences].

The following subsections will offer more detailed explanations for each method.

## Direct inversion

With `inversion_method = "direct"` pyDVL will calculate the inverse Hessian
using the direct matrix inversion. This means that the Hessian will first be
explicitly created and then inverted. This method is the most accurate, but also
the most computationally demanding. It is therefore not recommended for large
datasets or models with many parameters.

```python
import torch
from pydvl.influence.inversion import solve_hvp
b = torch.Tensor(...)
solve_hvp(
   "direct",
   wrapped_model,
   training_data_loader,
   b,
)
```

The result, an object of type 
[InverseHvpResult][pydvl.influence.framework.InverseHvpResult],
which holds two objects: `influence_factors.x` and `influence_factors.info`.
The first one is the inverse Hessian vector product, while the second one is a
dictionary with the info on the inversion process. For this method, the info
consists of the Hessian matrix itself.

## Conjugate Gradient

A classical method for solving linear systems of equations is the conjugate
gradient method. It is an iterative method that does not require the explicit
inversion of the Hessian matrix. Instead, it only requires the calculation of
the Hessian vector product. This makes it a good choice for large datasets or
models with many parameters. It is Nevertheless much slower than the direct
inversion method and not as accurate.
More info on the theory of conjugate gradient can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

In pyDVL, you can select conjugate gradient with `inversion_method = "cg"`, like
this:

```python
from pydvl.influence.inversion import solve_hvp
solve_hvp(
   "cg",
   wrapped_model,
   training_data_loader,
   b,
   x0=None,
   rtol=1e-7,
   atol=1e-7,
   maxiter=None,
)
```

The additional optional parameters `x0`, `rtol`, `atol`, and `maxiter` are passed
to the [solve_batch_cg][pydvl.influence.frameworks.torch_differentiable.solve_batch_cg]
function, and are respecively the initial guess for the solution, the relative
tolerance, the absolute tolerance, and the maximum number of iterations.

The resulting [InverseHvpResult][pydvl.influence.framework.InverseHvpResult]
holds the solution of the iHVP, `influence_factors.x`, and some info on the
inversion process `influence_factors.info`. More specifically, for each batch
the infos will report the number of iterations, a boolean indicating if the
inversion converged, and the residual of the inversion.

## Linear time Stochastic Second-Order Approximation (LiSSA)

The LiSSA method is a stochastic approximation of the inverse Hessian vector
product. Compared to [conjugate gradient](#conjugate-gradient)
it is faster but less accurate and typically suffers from 
instability.

In order to find the solution of the HVP, LiSSA iteratively approximates the
inverse of the Hessian matrix with the following update:

$$H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},$$

where $d$ and $s$ are a dampening and a scaling factor, which are essential
for the convergence of the method and they need to be chosen carefully, and I 
is the identity matrix. More info on the theory of LiSSA can be found in the 
original paper [@agarwal_2017_second].

In pyDVL, you can select LiSSA with `inversion_method = "lissa"`, like this:

```python
from pydvl.influence.inversion import solve_hvp
solve_hvp(
   "lissa",
   wrapped_model,
   training_data_loader,
   b,
   maxiter=1000,
   dampen=0.0,
   scale=10.0,
   h0=None,
   rtol=1e-4,
)
```

with the additional optional parameters `maxiter`, `dampen`, `scale`, `h0`, and
`rtol`, which are passed to the
[solve_lissa][pydvl.influence.frameworks.torch_differentiable.solve_lissa] function,
being the maximum number of iterations, the dampening factor, the scaling
factor, the initial guess for the solution and the relative tolerance,
respectively.

The resulting [InverseHvpResult][pydvl.influence.framework.InverseHvpResult]
holds the solution of the iHVP, `influence_factors.x`, and,
within `influence_factors.info`, the maximum percentage error
and the mean percentage error of the approximation.

## Arnoldi solver

The Arnoldi method is a Krylov subspace method for approximating the action of a
matrix on a vector. It is a generalization of the power method for finding
eigenvectors of a matrix.
