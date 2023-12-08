In almost every practical application it is not possible to construct, even less
invert the complete Hessian in memory. pyDVL offers several implementations of the interface
[InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel], which do not compute
the full Hessian (in contrast to [DirectInfluence][pydvl.influence.torch.influence_model.DirectInfluence]).


#### Conjugate Gradient

This classical procedure for solving linear systems of equations is an iterative
method that does not require the explicit inversion of the Hessian. Instead, it
only requires the calculation of Hessian-vector products, making it a good
choice for large datasets or models with many parameters. It is nevertheless
much slower to converge than the direct inversion method and not as accurate.
More info on the theory of conjugate gradient can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

```python
from pydvl.influence.torch import CgInfluence

if_model = CgInfluence(
    model,
    loss,
    hessian_regularization=0.0,
    x0=None,
    rtol=1e-7,
    atol=1e-7,
    maxiter=None,
)
```

The additional optional parameters `x0`, `rtol`, `atol`, and `maxiter` are
respectively the initial guess for the solution, the relative
tolerance, the absolute tolerance, and the maximum number of iterations.


#### Linear time Stochastic Second-Order Approximation (LiSSA)

The LiSSA method is a stochastic approximation of the inverse Hessian vector
product. Compared to [conjugate gradient](#conjugate-gradient)
it is faster but less accurate and typically suffers from instability.

In order to find the solution of the HVP, LiSSA iteratively approximates the
inverse of the Hessian matrix with the following update:

$$H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},$$

where $d$ and $s$ are a dampening and a scaling factor, which are essential
for the convergence of the method and they need to be chosen carefully, and I
is the identity matrix. More info on the theory of LiSSA can be found in the
original paper [@agarwal_secondorder_2017].


```python
from pydvl.influence.torch import LissaInfluence
if_model = LissaInfluence(
   model,
   loss,
   hessian_regularization=0.0 
   maxiter=1000,
   dampen=0.0,
   scale=10.0,
   h0=None,
   rtol=1e-4,
)
```

with the additional optional parameters `maxiter`, `dampen`, `scale`, `h0`, and
`rtol`,
being the maximum number of iterations, the dampening factor, the scaling
factor, the initial guess for the solution and the relative tolerance,
respectively.

#### Arnoldi

The [Arnoldi method](https://en.wikipedia.org/wiki/Arnoldi_iteration) is a
Krylov subspace method for approximating dominating eigenvalues and
eigenvectors. Under a low rank assumption on the Hessian at a minimizer (which
is typically observed for deep neural networks), this approximation captures the
essential action of the Hessian. More concretely, for $Hx=b$ the solution is
approximated by

\[x \approx V D^{-1} V^T b\]

where \(D\) is a diagonal matrix with the top (in absolute value) eigenvalues of
the Hessian and \(V\) contains the corresponding eigenvectors. See also
[@schioppa_scaling_2021].

```python
from pydvl.influence.torch import ArnoldiInfluence
if_model = ArnoldiInfluence
    model,
    loss,
    hessian_regularization=0.0,
    rank_estimate=10,
    tol=1e-6,
)
```
These implementations represent the calculation logic on in memory tensors. To scale up to large collection
of data, we map these influence function models over these collections. For a detailed discussion see the
documentation page [Scaling Computation](scaling_computation.md).