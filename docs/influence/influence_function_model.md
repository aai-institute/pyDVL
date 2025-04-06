In almost every practical application it is not possible to construct, even less
invert the complete Hessian in memory. pyDVL offers several implementations of 
the interface [InfluenceFunctionModel
][pydvl.influence.base_influence_function_model.InfluenceFunctionModel], 
which do not compute the full Hessian (in contrast to [DirectInfluence
][pydvl.influence.torch.influence_function_model.DirectInfluence]).


### Conjugate Gradient

This classical procedure for solving linear systems of equations is an iterative
method that does not require the explicit inversion of the Hessian. Instead, it
only requires the calculation of Hessian-vector products, making it a good
choice for large datasets or models with many parameters. It is nevertheless
much slower to converge than the direct inversion method and not as accurate.

More info on the theory of conjugate gradient can be found on
[Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method), or in
text books such as [@trefethen_numerical_1997, Lecture 38].

pyDVL also implements a stable block variant of the conjugate 
gradient method, defined in [@ji_breakdownfree_2017], which solves several
right hand sides simultaneously.

Optionally, the user can provide a pre-conditioner to improve convergence, such 
as a [Jacobi preconditioner
][pydvl.influence.torch.preconditioner.JacobiPreconditioner], which
is a simple [diagonal pre-conditioner](
https://en.wikipedia.org/wiki/Preconditioner#Jacobi_(or_diagonal)_preconditioner) 
based on Hutchinson's diagonal estimator [@bekas_estimator_2007],
or a [Nyström approximation based preconditioner
][pydvl.influence.torch.preconditioner.NystroemPreconditioner], 
described in [@frangella_randomized_2023].

??? Example "Using Conjugate Gradient"
  ```python
  from pydvl.influence.torch import CgInfluence, BlockMode, SecondOrderMode
  from pydvl.influence.torch.preconditioner import NystroemPreconditioner
  
  if_model = CgInfluence(
      model,
      loss,
      regularization=0.0,
      rtol=1e-7,
      atol=1e-7,
      maxiter=None,
      solve_simultaneously=True,
      preconditioner=NystroemPreconditioner(rank=10),
      block_structure=BlockMode.FULL,
      second_order_mode=SecondOrderMode.HESSIAN
  )
  if_model.fit(train_loader)
  ```

The additional optional parameters `rtol`, `atol`, `maxiter`, 
`solve_simultaneously` and `preconditioner` are respectively, the relative
tolerance, the absolute tolerance, the maximum number of iterations, 
a flag indicating whether to use a variant of cg to
simultaneously solving the system for several right hand sides and an optional
preconditioner.

This implementation is capable of using a block-diagonal
approximation, see
[Block-diagonal approximation][block-diagonal-approximation], and can handle
[Gauss-Newton approximation][gauss-newton-approximation].


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

??? Example "Using LiSSA"
  ```python
  from pydvl.influence.torch import LissaInfluence, BlockMode, SecondOrderMode
  if_model = LissaInfluence(
     model,
     loss,
     regularization=0.0, 
     maxiter=1000,
     dampen=0.0,
     scale=10.0,
     rtol=1e-4,
     block_structure=BlockMode.FULL,
     second_order_mode=SecondOrderMode.GAUSS_NEWTON
  )
  if_model.fit(train_loader)
  ```

with the additional optional parameters `maxiter`, `dampen`, `scale`, and
`rtol`,
being the maximum number of iterations, the dampening factor, the scaling
factor and the relative tolerance,
respectively. This implementation is capable of using a block-matrix 
approximation, see 
[Block-diagonal approximation][block-diagonal-approximation], and can handle
[Gauss-Newton approximation][gauss-newton-approximation].

### Arnoldi { #arnoldi-method }

The [Arnoldi method](https://en.wikipedia.org/wiki/Arnoldi_iteration) is a
Krylov subspace method for approximating dominating eigenvalues and
eigenvectors. Under a low rank assumption on the Hessian at a minimizer (which
is typically observed for deep neural networks), this approximation captures the
essential action of the Hessian. More concretely, for $Hx=b$ the solution is
approximated by

\[x \approx V D^{-1} V^T b\]

where \(D\) is a diagonal matrix with the top (in absolute value) eigenvalues of
the Hessian and \(V\) contains the corresponding eigenvectors. See also
[@schioppa_scaling_2022].

??? Example "Using Arnoldi"
  ```python
  from pydvl.influence.torch import ArnoldiInfluence, BlockMode, SecondOrderMode
  if_model = ArnoldiInfluence(
      model,
      loss,
      regularization=0.0,
      rank=10,
      tol=1e-6,
      block_structure=BlockMode.FULL,
      second_order_mode=SecondOrderMode.HESSIAN
  )
  if_model.fit(train_loader)
  ```

This implementation is capable of using a block-matrix
approximation, see
[Block-diagonal approximation][block-diagonal-approximation], and can handle
[Gauss-Newton approximation][gauss-newton-approximation].

### Eigenvalue Corrected K-FAC

K-FAC, short for Kronecker-Factored Approximate Curvature, is a method that 
approximates the Fisher information matrix [FIM](https://en.wikipedia.org/wiki/Fisher_information) of a model. 
It is possible to show that for classification models with appropriate loss 
functions the FIM is equal to the Hessian of the model’s loss over the dataset. 
In this restricted but nonetheless important context K-FAC offers an efficient 
way to approximate the Hessian and hence the influence scores. 
For more info and details refer to the original paper [@martens_optimizing_2015].

The K-FAC method is implemented in the class [EkfacInfluence
][pydvl.influence.torch.influence_function_model.EkfacInfluence]. 
The following code snippet shows how to use the K-FAC method to calculate the 
influence function of a model. Note that, in contrast to the other methods for 
influence function calculation, K-FAC does not require the loss function as an 
input. This is because the current implementation is only applicable to 
classification models with a cross entropy loss function. 

??? Example " Using K-FAC"
    ```python
    from pydvl.influence.torch import EkfacInfluence
    if_model = EkfacInfluence(
        model,
        hessian_regularization=0.0,
    )
    if_model.fit(train_loader)
    ```

Upon initialization, the K-FAC method will parse the model and extract which 
layers require grad and which do not. Then it will only calculate the influence 
scores for the layers that require grad. The current implementation of the 
K-FAC method is only available for linear layers, and therefore if the model 
contains non-linear layers that require gradient the K-FAC method will raise a 
NotImplementedLayerRepresentationException.

A further improvement of the K-FAC method is the Eigenvalue Corrected 
K-FAC (EKFAC) method [@george_fast_2018], which allows to further re-fit the 
eigenvalues of the Hessian, thus providing a more accurate approximation. 
On top of the K-FAC method, the EKFAC method is implemented by setting 
`update_diagonal=True` when initialising [EkfacInfluence
][pydvl.influence.torch.influence_function_model.EkfacInfluence]. 


??? Example "Using EKFAC"
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
H_{\text{nys}} &= (H\Omega)(\Omega^TH\Omega)^{\dagger}(H\Omega)^T \\\
&= U \Lambda U^T,
\end{align*}

where $(\cdot)^{\dagger}$ denotes the [Moore-Penrose inverse](
https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
in combination with the [Sherman–Morrison–Woodbury formula](
https://en.wikipedia.org/wiki/Woodbury_matrix_identity) to calculate the
action of its inverse:

\begin{equation*} 
(H_{\text{nys}} + \lambda I)^{-1}x = U(\Lambda+\lambda I)U^Tx +
\frac{1}{\lambda}(I−UU^T)x,
\end{equation*}

see also [@hataya_nystrom_2023] and [@frangella_randomized_2023]. The essential 
parameter is the rank of the approximation.


??? Example "Using Nyström Sketch-and-Solve"
  ```python
  from pydvl.influence.torch import NystroemSketchInfluence, BlockMode, SecondOrderMode
  if_model = NystroemSketchInfluence(
      model,
      loss,
      rank=10,
      regularization=0.0,
      block_structure=BlockMode.FULL,
      second_order_mode=SecondOrderMode.HESSIAN
  )
  if_model.fit(train_loader)
  ```

This implementation is capable of using a block-matrix approximation, see
[Block-diagonal approximation][block-diagonal-approximation], and can handle
[Gauss-Newton approximation][gauss-newton-approximation].

### Inverse Harmonic Mean

This implementation replaces the inverse Hessian matrix in the influence computation
with an approximation of the inverse Gauss-Newton vector product and was
proposed in [@kwon_datainf_2023].

The approximation method comprises
the following steps:

1.  Replace the Hessian $H(\theta)$ with the Gauss-Newton matrix 
    $G(\theta)$:
    
    \begin{equation*}
        G(\theta)=n^{-1} \sum_{i=1}^n \nabla_{\theta}\ell_i\nabla_{\theta}\ell_i^T
    \end{equation*}
    
    which results in

    \begin{equation*}
        \mathcal{I}(z_{t}, z) \approx \nabla_{\theta} \ell(z_{t}, \theta)^T 
                         (G(\theta) + \lambda I_d)^{-1} 
                         \nabla_{\theta} \ell(z, \theta) 
    \end{equation*}

2.  Simplify the problem by breaking it down into a block diagonal structure, 
    where each block $G_l(\theta)$ corresponds to the l-th block:   
    
    \begin{equation*}
        G_{l}(\theta) = n^{-1} \sum_{i=1}^n \nabla_{\theta_l} \ell_i 
                       \nabla_{\theta_l} \ell_i^{T} + \lambda_l I_{d_l},
    \end{equation*}
       
    which leads to
       
    \begin{equation*}
       \mathcal{I}(z_{t}, z) \approx \nabla_{\theta} \ell(z_{t}, \theta)^T 
                                     \operatorname{diag}(G_1(\theta)^{-1}, 
                                     \dots, G_L(\theta)^{-1}) 
                                     \nabla_{\theta} \ell(z, \theta)
    \end{equation*}

3.  Substitute the arithmetic mean of the rank-$1$ updates in 
       $G_l(\theta)$, with the inverse harmonic mean $R_l(\theta)$ of the rank-1 
    updates:
       
    \begin{align*}
        G_l(\theta)^{-1} &= \left(  n^{-1} \sum_{i=1}^n \nabla_{\theta_l} 
                           \ell(z_i, \theta) \nabla_{\theta_l} 
                           \ell(z_i, \theta)^{T} + 
                           \lambda_l I_{d_l}\right)^{-1} \\\
        R_{l}(\theta)&= n^{-1} \sum_{i=1}^n \left( \nabla_{\theta_l} 
                       \ell(z_i, \theta) \nabla_{\theta_l} \ell(z_i, \theta)^{T} 
                       + \lambda_l I_{d_l} \right)^{-1}
    \end{align*}

4.  Use the 
   <a href="https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula">
     Sherman–Morrison formula
   </a> 
    to get an explicit representation of the inverses in the definition of 
    $R_l(\theta):$
    
    \begin{align*}
        R_l(\theta) &= n^{-1} \sum_{i=1}^n \left( \nabla_{\theta_l} \ell_i
        \nabla_{\theta_l} \ell_i^{T}
        + \lambda_l I_{d_l}\right)^{-1} \\\
        &= n^{-1} \sum_{i=1}^n \lambda_l^{-1} \left(I_{d_l}
        - \frac{\nabla_{\theta_l} \ell_i \nabla_{\theta_l}
        \ell_i^{T}}{\lambda_l
        + \\|\nabla_{\theta_l} \ell_i\\|_2^2}\right)
        ,
    \end{align*}

    which means application of $R_l(\theta)$ boils down to computing $n$
    rank-$1$ updates.

```python
from pydvl.influence.torch import InverseHarmonicMeanInfluence, BlockMode

if_model = InverseHarmonicMeanInfluence(
    model,
    loss,
    regularization=1e-1,
    block_structure=BlockMode.LAYER_WISE
)
if_model.fit(train_loader)
```
This implementation is capable of using a block-matrix approximation, see
[Block-diagonal approximation][block-diagonal-approximation].


These implementations represent the calculation logic on in memory tensors. 
To scale up to large collection of data, we map these influence function models 
over these collections. For a detailed discussion see the
documentation page [Scaling Computation](scaling_computation.md).


