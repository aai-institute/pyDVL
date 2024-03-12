In almost every practical application it is not possible to construct, even less
invert the complete Hessian in memory. pyDVL offers several implementations of the interface
[InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel], which do not compute
the full Hessian (in contrast to [DirectInfluence][pydvl.influence.torch.influence_function_model.DirectInfluence]).


### Conjugate Gradient

This classical procedure for solving linear systems of equations is an iterative
method that does not require the explicit inversion of the Hessian. Instead, it
only requires the calculation of Hessian-vector products, making it a good
choice for large datasets or models with many parameters. It is nevertheless
much slower to converge than the direct inversion method and not as accurate.

More info on the theory of conjugate gradient can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method). 

pyDVL also implements a stable block variant of the conjugate 
gradient method, defined in [@ji_breakdownfree_2017], which solves several
right hand sides simultaneously.

Optionally, the user can provide a pre-conditioner to improve convergence, such as
a [Jacobi pre-conditioner](
https://en.wikipedia.org/wiki/Preconditioner#Jacobi_(or_diagonal)_preconditioner)
or a Nyström approximation based pre-conditioner, 
described in [@frangella_randomized_2023]. 

```python
from pydvl.influence.torch import CgInfluence
from pydvl.influence.torch.pre_conditioner import NystroemPreConditioner

if_model = CgInfluence(
    model,
    loss,
    hessian_regularization=0.0,
    rtol=1e-7,
    atol=1e-7,
    maxiter=None,
    use_block_cg=True,
    pre_conditioner=NystroemPreConditioner(rank=10)
)
if_model.fit(train_loader)
```

The additional optional parameters `rtol`, `atol`, `maxiter`, `use_block_cg` and 
`pre_conditioner` are respectively, the relative
tolerance, the absolute tolerance, the maximum number of iterations, 
a flag indicating whether to use block variant of cg and an optional
pre-conditioner.


### Linear time Stochastic Second-Order Approximation (LiSSA)

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
if_model.fit(train_loader)
```

with the additional optional parameters `maxiter`, `dampen`, `scale`, `h0`, and
`rtol`,
being the maximum number of iterations, the dampening factor, the scaling
factor, the initial guess for the solution and the relative tolerance,
respectively.

### Arnoldi

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
if_model = ArnoldiInfluence(
    model,
    loss,
    hessian_regularization=0.0,
    rank_estimate=10,
    tol=1e-6,
)
if_model.fit(train_loader)
```

### Eigenvalue Corrected K-FAC

K-FAC, short for Kronecker-Factored Approximate Curvature, is a method that approximates the Fisher information matrix [FIM](https://en.wikipedia.org/wiki/Fisher_information) of a model. It is possible to show that for classification models with appropriate loss functions the FIM is equal to the Hessian of the model’s loss over the dataset. In this restricted but nonetheless important context K-FAC offers an efficient way to approximate the Hessian and hence the influence scores. 
For more info and details refer to the original paper [@martens_optimizing_2015].

The K-FAC method is implemented in the class [EkfacInfluence](pydvl/influence/torch/influence_function_model.py). The following code snippet shows how to use the K-FAC method to calculate the influence function of a model. Note that, in contrast to the other methods for influence function calculation, K-FAC does not require the loss function as an input. This is because the current implementation is only applicable to classification models with a cross entropy loss function. 

```python
from pydvl.influence.torch import EkfacInfluence
if_model = EkfacInfluence(
    model,
    hessian_regularization=0.0,
)
if_model.fit(train_loader)
```
Upon initialization, the K-FAC method will parse the model and extract which layers require grad and which do not. Then it will only calculate the influence scores for the layers that require grad. The current implementation of the K-FAC method is only available for linear layers, and therefore if the model contains non-linear layers that require gradient the K-FAC method will raise a NotImplementedLayerRepresentationException.

A further improvement of the K-FAC method is the Eigenvalue Corrected K-FAC (EKFAC) method [@george_fast_2018], which allows to further re-fit the eigenvalues of the Hessian, thus providing a more accurate approximation. On top of the K-FAC method, the EKFAC method is implemented by setting `update_diagonal=True` when initialising [EkfacInfluence](pydvl/influence/torch/influence_function_model.py). The following code snippet shows how to use the EKFAC method to calculate the influence function of a model. 

```python
from pydvl.influence.torch import EkfacInfluence
if_model = EkfacInfluence(
    model,
    update_diagonal=True,
    hessian_regularization=0.0,
)
if_model.fit(train_loader)
```

### Nyström Sketch-and-Solve

This approximation is based on a Nyström low-rank approximation of the form

\begin{align*}
H_{\text{nys}} &= (H\Omega)(\Omega^TH\Omega)^{+}(H\Omega)^T \\\
&= U \Lambda U^T
\end{align*}

in combination with the [Sherman–Morrison–Woodbury formula](
https://en.wikipedia.org/wiki/Woodbury_matrix_identity) to calculate the
action of its inverse:

\begin{equation*} 
(H_{\text{nys}} + \lambda I)^{-1}x = U(\Lambda+\lambda I)U^Tx +
\frac{1}{\lambda}(I−UU^T)x,
\end{equation*}

see also [@hataya_nystrom_2023] and [@frangella_randomized_2021]. The essential parameter is the rank of the
approximation.

```python
from pydvl.influence.torch import NystroemSketchInfluence
if_model = NystroemSketchInfluence(
    model,
    loss,
    rank=10,
    hessian_regularization=0.0,
)
if_model.fit(train_loader)
```

These implementations represent the calculation logic on in memory tensors. To scale up to large collection
of data, we map these influence function models over these collections. For a detailed discussion see the
documentation page [Scaling Computation](scaling_computation.md).
