"""
Contains methods for differentiating  a pyTorch model. Most of the methods focus
on ways to calculate matrix vector products. Moreover, it contains several
methods to invert the Hessian vector product. These are used to calculate the
influence of a training point on the model.
"""
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ...utils import maybe_progress
from ..inversion import InversionMethod, InversionRegistry
from ..twice_differentiable import (
    InverseHvpResult,
    TensorUtilities,
    TwiceDifferentiable,
)
from .functional import get_hvp_function
from .util import align_structure, as_tensor, flatten_tensors_to_vector

__all__ = [
    "TorchTwiceDifferentiable",
    "solve_linear",
    "solve_batch_cg",
    "solve_lissa",
    "solve_arnoldi",
    "lanzcos_low_rank_hessian_approx",
    "as_tensor",
    "model_hessian_low_rank",
]

logger = logging.getLogger(__name__)


class TorchTwiceDifferentiable(TwiceDifferentiable[torch.Tensor]):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        r"""
        :param model: A (differentiable) function.
        :param loss:  A differentiable scalar loss $L(\hat{y}, y)$,
            mapping a prediction and a target to a real value.
        """
        if model.training:
            logger.warning(
                "Passed model not in evaluation mode. This can create several issues in influence "
                "computation, e.g. due to batch normalization. Please call model.eval() before "
                "computing influences."
            )
        self.loss = loss
        self.model = model
        first_param = next(model.parameters())
        self.device = first_param.device
        self.dtype = first_param.dtype

    @classmethod
    def tensor_type(cls):
        return torch.Tensor

    @property
    def parameters(self) -> List[torch.Tensor]:
        """Returns all the model parameters that require differentiating"""
        return [param for param in self.model.parameters() if param.requires_grad]

    @property
    def num_params(self) -> int:
        """
        Get number of parameters of model f.
        :returns: Number of parameters as integer.
        """
        return sum([p.numel() for p in self.parameters])

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """
        Calculates gradient of model parameters wrt the model parameters.

        :param x: A matrix [NxD] representing the features $x_i$.
        :param y: A matrix [NxK] representing the target values $y_i$.
        :param create_graph: If True, the resulting gradient tensor, can be used for further differentiation
        :returns: An array [P] with the gradients of the model.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        if create_graph and not x.requires_grad:
            x = x.requires_grad_(True)

        loss_value = self.loss(torch.squeeze(self.model(x)), torch.squeeze(y))
        grad_f = torch.autograd.grad(
            loss_value, self.parameters, create_graph=create_graph
        )
        return flatten_tensors_to_vector(grad_f)

    def hessian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the explicit hessian of model parameters given data ($x$ and $y$).
        :param x: A matrix [NxD] representing the features $x_i$.
        :param y: A matrix [NxK] representing the target values $y_i$.
        :returns: A tensor representing the hessian of the loss wrt. the model parameters.
        """

        def model_func(param):
            outputs = torch.func.functional_call(
                self.model,
                align_structure(
                    {k: p for k, p in self.model.named_parameters() if p.requires_grad},
                    param,
                ),
                (x.to(self.device),),
                strict=True,
            )
            return self.loss(outputs, y.to(self.device))

        params = flatten_tensors_to_vector(
            p.detach() for p in self.model.parameters() if p.requires_grad
        )
        return torch.func.hessian(model_func)(params)

    @staticmethod
    def mvp(
        grad_xy: torch.Tensor,
        v: torch.Tensor,
        backprop_on: torch.Tensor,
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Calculates second order derivative of the model along directions v.
        This second order derivative can be selected through the backprop_on argument.

        :param grad_xy: an array [P] holding the gradients of the model
            parameters wrt input $x$ and labels $y$, where P is the number of
            parameters of the model. It is typically obtained through
            self.grad.
        :param v: An array ([DxP] or even one dimensional [D]) which
            multiplies the matrix, where D is the number of directions.
        :param progress: True, iff progress shall be printed.
        :param backprop_on: tensor used in the second backpropagation (the first
            one is along $x$ and $y$ as defined via grad_xy).
        :returns: A matrix representing the implicit matrix vector product
            of the model along the given directions. Output shape is [DxP] if
            backprop_on is None, otherwise [DxM], with M the number of elements
            of backprop_on.
        """
        device = grad_xy.device
        v = as_tensor(v, warn=False).to(device)
        if v.ndim == 1:
            v = v.unsqueeze(0)

        z = (grad_xy * Variable(v)).sum(dim=1)

        mvp = []
        for i in maybe_progress(range(len(z)), progress, desc="MVP"):
            mvp.append(
                flatten_tensors_to_vector(
                    autograd.grad(z[i], backprop_on, retain_graph=True)
                )
            )
        return torch.stack([grad.contiguous().view(-1) for grad in mvp]).detach()


@dataclass
class LowRankProductRepresentation:
    """
    Representation of a low rank product of the form $H = V D V^T$, where D is a diagonal matrix and
    V is orthogonal
    :param eigen_vals: diagonal of D
    :param projections: the matrix V
    """

    eigen_vals: torch.Tensor
    projections: torch.Tensor

    @property
    def device(self) -> torch.device:
        return (
            self.eigen_vals.device
            if hasattr(self.eigen_vals, "device")
            else torch.device("cpu")
        )

    def to(self, device: torch.device):
        """
        Move the representing tensors to a device
        """
        return LowRankProductRepresentation(
            self.eigen_vals.to(device), self.projections.to(device)
        )

    def __post_init__(self):
        if self.eigen_vals.device != self.projections.device:
            raise ValueError("eigen_vals and projections must be on the same device.")


def lanzcos_low_rank_hessian_approx(
    hessian_vp: Callable[[torch.Tensor], torch.Tensor],
    matrix_shape: Tuple[int, int],
    hessian_perturbation: float = 0.0,
    rank_estimate: int = 10,
    krylov_dimension: Optional[int] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    device: Optional[torch.device] = None,
    eigen_computation_on_gpu: bool = False,
    torch_dtype: torch.dtype = None,
) -> LowRankProductRepresentation:
    """
    Calculates a low-rank approximation of the Hessian matrix of a scalar-valued
    function using the implicitly restarted Lanczos algorithm.

    :param hessian_vp: A function that takes a vector and returns the product of
        the Hessian of the loss function.
    :param matrix_shape: The shape of the matrix, represented by hessian vector
        product.
    :param hessian_perturbation: Optional regularization parameter added to the
        Hessian-vector product for numerical stability.
    :param rank_estimate: The number of eigenvalues and corresponding eigenvectors
        to compute. Represents the desired rank of the Hessian approximation.
    :param krylov_dimension: The number of Krylov vectors to use for the Lanczos
        method. If not provided, it defaults to
        $min(model.num_parameters, max(2*rank_estimate + 1, 20))$.
    :param tol: The stopping criteria for the Lanczos algorithm, which stops when
        the difference in the approximated eigenvalue is less than ``tol``.
        Defaults to 1e-6.
    :param max_iter: The maximum number of iterations for the Lanczos method. If
        not provided, it defaults to ``10 * model.num_parameters``.
    :param device: The device to use for executing the hessian vector product.
    :param eigen_computation_on_gpu: If ``True``, tries to execute the eigen pair
        approximation on the provided device via `cupy <https://cupy.dev/>`_
        implementation. Make sure that either your model is small enough, or you
        use a small rank_estimate to fit your device's memory. If ``False``, the
        eigen pair approximation is executed on the CPU with scipy's wrapper to
        ARPACK.
    :param torch_dtype: if not provided, current torch default dtype is used for
        conversion to torch.

    :return: An object that contains the top- ``rank_estimate`` eigenvalues and
        corresponding eigenvectors of the Hessian.
    """

    torch_dtype = torch.get_default_dtype() if torch_dtype is None else torch_dtype

    if eigen_computation_on_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
            from torch.utils.dlpack import from_dlpack, to_dlpack
        except ImportError as e:
            raise ImportError(
                f"Try to install missing dependencies or set eigen_computation_on_gpu to False: {e}"
            )

        if device is None:
            raise ValueError(
                "Without setting an explicit device, cupy is not supported"
            )

        def to_torch_conversion_function(x):
            return from_dlpack(x.toDlpack()).to(torch_dtype)

        def mv(x):
            x = to_torch_conversion_function(x)
            y = hessian_vp(x) + hessian_perturbation * x
            return cp.from_dlpack(to_dlpack(y))

    else:
        from scipy.sparse.linalg import LinearOperator, eigsh

        def mv(x):
            x_torch = torch.as_tensor(x, device=device, dtype=torch_dtype)
            y: NDArray = (
                (hessian_vp(x_torch) + hessian_perturbation * x_torch)
                .detach()
                .cpu()
                .numpy()
            )
            return y

        to_torch_conversion_function = partial(torch.as_tensor, dtype=torch_dtype)

    try:
        eigen_vals, eigen_vecs = eigsh(
            LinearOperator(matrix_shape, matvec=mv),
            k=rank_estimate,
            maxiter=max_iter,
            tol=tol,
            ncv=krylov_dimension,
            return_eigenvectors=True,
        )

    except ArpackNoConvergence as e:
        logger.warning(
            f"ARPACK did not converge for parameters {max_iter=}, {tol=}, {krylov_dimension=}, "
            f"{rank_estimate=}. \n Returning the best approximation found so far. Use those with care or "
            f"modify parameters.\n Original error: {e}"
        )

        eigen_vals, eigen_vecs = e.eigenvalues, e.eigenvectors

    eigen_vals = to_torch_conversion_function(eigen_vals)
    eigen_vecs = to_torch_conversion_function(eigen_vecs)

    return LowRankProductRepresentation(eigen_vals, eigen_vecs)


def model_hessian_low_rank(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    hessian_perturbation: float = 0.0,
    rank_estimate: int = 10,
    krylov_dimension: Optional[int] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    eigen_computation_on_gpu: bool = False,
) -> LowRankProductRepresentation:
    """
    Calculates a low-rank approximation of the Hessian matrix of the model's loss function using the implicitly
    restarted Lanczos algorithm.

    :param model: A PyTorch model instance that is twice differentiable, wrapped into :class:`TorchTwiceDifferential`.
                  The Hessian will be calculated with respect to this model's parameters.
    :param training_data: A DataLoader instance that provides the model's training data.
                          Used in calculating the Hessian-vector products.
    :param hessian_perturbation: Optional regularization parameter added to the Hessian-vector product
                                 for numerical stability.
    :param rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
                          Represents the desired rank of the Hessian approximation.
    :param krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
                             If not provided, it defaults to $min(model.num_parameters, max(2*rank_estimate + 1, 20))$.
    :param tol: The stopping criteria for the Lanczos algorithm, which stops when the difference
                in the approximated eigenvalue is less than `tol`. Defaults to 1e-6.
    :param max_iter: The maximum number of iterations for the Lanczos method. If not provided, it defaults to
                     $10*model.num_parameters$
    :param eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the provided
                                     device via cupy implementation.
                                     Make sure, that either your model is small enough or you use a
                                     small rank_estimate to fit your device's memory.
                                     If False, the eigen pair approximation is executed on the CPU by scipy wrapper to
                                     ARPACK.
    :return: A `LowRankProductRepresentation` instance that contains the top (up until rank_estimate) eigenvalues
             and corresponding eigenvectors of the Hessian.
    """
    raw_hvp = get_hvp_function(
        model.model, model.loss, training_data, use_hessian_avg=True
    )

    return lanzcos_low_rank_hessian_approx(
        hessian_vp=raw_hvp,
        matrix_shape=(model.num_params, model.num_params),
        hessian_perturbation=hessian_perturbation,
        rank_estimate=rank_estimate,
        krylov_dimension=krylov_dimension,
        tol=tol,
        max_iter=max_iter,
        device=model.device if hasattr(model, "device") else None,
        eigen_computation_on_gpu=eigen_computation_on_gpu,
    )


class TorchTensorUtilities(TensorUtilities[torch.Tensor, TorchTwiceDifferentiable]):
    twice_differentiable_type = TorchTwiceDifferentiable

    @staticmethod
    def einsum(equation: str, *operands) -> torch.Tensor:
        """Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
        based on the Einstein summation convention.
        """
        return torch.einsum(equation, *operands)

    @staticmethod
    def cat(a: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        """Concatenates a sequence of tensors into a single torch tensor"""
        return torch.cat(a, **kwargs)

    @staticmethod
    def stack(a: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        """Stacks a sequence of tensors into a single torch tensor"""
        return torch.stack(a, **kwargs)

    @staticmethod
    def unsqueeze(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Add a singleton dimension at a specified position in a tensor.

        :param x: A PyTorch tensor.
        :param dim: The position at which to add the singleton dimension. Zero-based indexing.
        :return: A new tensor with an additional singleton dimension.
        """
        return x.unsqueeze(dim)

    @staticmethod
    def get_element(x: torch.Tensor, idx: int) -> torch.Tensor:
        return x[idx]

    @staticmethod
    def slice(x: torch.Tensor, start: int, stop: int, axis: int = 0) -> torch.Tensor:
        slicer = [slice(None) for _ in x.shape]
        slicer[axis] = slice(start, stop)
        return x[tuple(slicer)]

    @staticmethod
    def shape(x: torch.Tensor) -> Tuple[int, ...]:
        return x.shape  # type:ignore

    @staticmethod
    def reshape(x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        return x.reshape(shape)

    @staticmethod
    def cat_gen(
        a: Generator[torch.Tensor, None, None],
        resulting_shape: Tuple[int, ...],
        model: TorchTwiceDifferentiable,
        axis: int = 0,
    ) -> torch.Tensor:
        result = torch.empty(resulting_shape, dtype=model.dtype, device=model.device)

        start_idx = 0
        for x in a:
            stop_idx = start_idx + x.shape[axis]

            slicer = [slice(None) for _ in resulting_shape]
            slicer[axis] = slice(start_idx, stop_idx)

            result[tuple(slicer)] = x

            start_idx = stop_idx

        return result


@InversionRegistry.register(TorchTwiceDifferentiable, InversionMethod.Direct)
def solve_linear(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    hessian_perturbation: float = 0.0,
) -> InverseHvpResult:
    """Given a model and training data, it finds x s.t. $Hx = b$, with $H$ being
    the model hessian.

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix, the right hand side of the equation $Hx = b$.
    :param hessian_perturbation: regularization of the hessian

    :return: An array that solves the inverse problem,
        i.e. it returns $x$ such that $Hx = b$, and a dictionary containing
        information about the solution.
    """

    all_x, all_y = [], []
    for x, y in training_data:
        all_x.append(x)
        all_y.append(y)
    hessian = model.hessian(torch.cat(all_x), torch.cat(all_y))
    matrix = hessian + hessian_perturbation * torch.eye(
        model.num_params, device=model.device
    )
    info = {"hessian": hessian}
    return InverseHvpResult(x=torch.linalg.solve(matrix, b.T).T, info=info)


@InversionRegistry.register(TorchTwiceDifferentiable, InversionMethod.Cg)
def solve_batch_cg(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    hessian_perturbation: float = 0.0,
    *,
    x0: Optional[torch.Tensor] = None,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    maxiter: Optional[int] = None,
    progress: bool = False,
) -> InverseHvpResult:
    """
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the Hessian Vector Product. More precisely, it finds x s.t. $Hx =
    b$, with $H$ being the model hessian. For more info, see
    `Wikipedia <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix, the right hand side of the equation $Hx = b$.
    :param hessian_perturbation: regularization of the hessian
    :param x0: initial guess for hvp. If None, defaults to b
    :param rtol: maximum relative tolerance of result
    :param atol: absolute tolerance of result
    :param maxiter: maximum number of iterations. If None, defaults to 10*len(b)
    :param progress: If True, display progress bars.

    :return: A matrix of shape [NxP] with each line being a solution of $Ax=b$,
        and a dictionary containing information about the convergence of CG, one
        entry for each line of the matrix.
    """
    total_grad_xy = 0
    total_points = 0
    for x, y in maybe_progress(training_data, progress, desc="Batch Train Gradients"):
        grad_xy = model.grad(x, y, create_graph=True)
        total_grad_xy += grad_xy * len(x)
        total_points += len(x)
    backprop_on = model.parameters
    reg_hvp = lambda v: model.mvp(
        total_grad_xy / total_points, v, backprop_on
    ) + hessian_perturbation * v.type(torch.float64)
    batch_cg = torch.zeros_like(b)
    info = {}
    for idx, bi in enumerate(maybe_progress(b, progress, desc="Conjugate gradient")):
        batch_result, batch_info = solve_cg(
            reg_hvp, bi, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter
        )
        batch_cg[idx] = batch_result
        info[f"batch_{idx}"] = batch_info
    return InverseHvpResult(x=batch_cg, info=info)


def solve_cg(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: Optional[torch.Tensor] = None,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    maxiter: Optional[int] = None,
) -> InverseHvpResult:
    """Conjugate gradient solver for the Hessian vector product

    :param hvp: a Callable Hvp, operating with tensors of size N
    :param b: a vector or matrix, the right hand side of the equation $Hx = b$.
    :param x0: initial guess for hvp
    :param rtol: maximum relative tolerance of result
    :param atol: absolute tolerance of result
    :param maxiter: maximum number of iterations. If None, defaults to 10*len(b)

    :return: A vector x, solution of $Ax=b$, and a dictionary containing
        information about the convergence of CG.
    """
    if x0 is None:
        x0 = torch.clone(b)
    if maxiter is None:
        maxiter = len(b) * 10

    y_norm = torch.sum(torch.matmul(b, b)).item()
    stopping_val = max([rtol**2 * y_norm, atol**2])

    x = x0
    p = r = (b - hvp(x)).squeeze().type(torch.float64)
    gamma = torch.sum(torch.matmul(r, r)).item()
    optimal = False

    for k in range(maxiter):
        if gamma < stopping_val:
            optimal = True
            break
        Ap = hvp(p).squeeze()
        alpha = gamma / torch.sum(torch.matmul(p, Ap)).item()
        x += alpha * p
        r -= alpha * Ap
        gamma_ = torch.sum(torch.matmul(r, r)).item()
        beta = gamma_ / gamma
        gamma = gamma_
        p = r + beta * p

    info = {"niter": k, "optimal": optimal, "gamma": gamma}
    return InverseHvpResult(x=x, info=info)


@InversionRegistry.register(TorchTwiceDifferentiable, InversionMethod.Lissa)
def solve_lissa(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    hessian_perturbation: float = 0.0,
    *,
    maxiter: int = 1000,
    dampen: float = 0.0,
    scale: float = 10.0,
    h0: Optional[torch.Tensor] = None,
    rtol: float = 1e-4,
    progress: bool = False,
) -> InverseHvpResult:
    r"""
    Uses LISSA, Linear time Stochastic Second-Order Algorithm, to iteratively
    approximate the inverse Hessian. More precisely, it finds x s.t. $Hx = b$,
    with $H$ being the model's second derivative wrt. the parameters.
    This is done with the update

    $$H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},$$

    where $I$ is the identity matrix, $d$ is a dampening term and $s$ a scaling
    factor that are applied to help convergence. For details, see
    :footcite:t:`koh_understanding_2017` and the original paper
    :footcite:t:`agarwal_2017_second`.

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix, the right hand side of the equation $Hx = b$.
    :param hessian_perturbation: regularization of the hessian
    :param progress: If True, display progress bars.
    :param maxiter: maximum number of iterations,
    :param dampen: dampening factor, defaults to 0 for no dampening
    :param scale: scaling factor, defaults to 10
    :param h0: initial guess for hvp

    :return: A matrix of shape [NxP] with each line being a solution of $Ax=b$,
        and a dictionary containing information about the accuracy of the solution.
    """
    if h0 is None:
        h_estimate = torch.clone(b)
    else:
        h_estimate = h0
    shuffled_training_data = DataLoader(
        training_data.dataset, training_data.batch_size, shuffle=True
    )

    def lissa_step(
        h: torch.Tensor, reg_hvp: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Given an estimate of the hessian inverse and the regularised hessian
        vector product, it computes the next estimate.

        :param h: an estimate of the hessian inverse
        :param reg_hvp: regularised hessian vector product
        :return: the next estimate of the hessian inverse
        """
        return b + (1 - dampen) * h - reg_hvp(h) / scale

    for _ in maybe_progress(range(maxiter), progress, desc="Lissa"):
        x, y = next(iter(shuffled_training_data))
        grad_xy = model.grad(x, y, create_graph=True)
        reg_hvp = (
            lambda v: model.mvp(grad_xy, v, model.parameters) + hessian_perturbation * v
        )
        residual = lissa_step(h_estimate, reg_hvp) - h_estimate
        h_estimate += residual
        if torch.isnan(h_estimate).any():
            raise RuntimeError("NaNs in h_estimate. Increase scale or dampening.")
        max_residual = torch.max(torch.abs(residual / h_estimate))
        if max_residual < rtol:
            break
    mean_residual = torch.mean(torch.abs(residual / h_estimate))
    logger.info(
        f"Terminated Lissa with {max_residual*100:.2f} % max residual."
        f" Mean residual: {mean_residual*100:.5f} %"
    )
    info = {
        "max_perc_residual": max_residual * 100,
        "mean_perc_residual": mean_residual * 100,
    }
    return InverseHvpResult(x=h_estimate / scale, info=info)


@InversionRegistry.register(TorchTwiceDifferentiable, InversionMethod.Arnoldi)
def solve_arnoldi(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    hessian_perturbation: float = 0.0,
    *,
    rank_estimate: int = 10,
    krylov_dimension: Optional[int] = None,
    low_rank_representation: Optional[LowRankProductRepresentation] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    eigen_computation_on_gpu: bool = False,
) -> InverseHvpResult:
    """
    Solves the linear system Hx = b, where H is the Hessian of the model's loss function and b is the given right-hand
    side vector. The Hessian is approximated using a low-rank representation.

    :param model: A PyTorch model instance that is twice differentiable, wrapped into :class:`TorchTwiceDifferential`.
                  The Hessian will be calculated with respect to this model's parameters.
    :param training_data: A DataLoader instance that provides the model's training data.
                          Used in calculating the Hessian-vector products.
    :param b: The right-hand side vector in the system Hx = b.
    :param hessian_perturbation: Optional regularization parameter added to the Hessian-vector product
                                 for numerical stability.
    :param rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
                          Represents the desired rank of the Hessian approximation.
    :param krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
                             If not provided, it defaults to $min(model.num_parameters, max(2*rank_estimate + 1, 20))$.
    :param low_rank_representation: A LowRankProductRepresentation instance containing a previously computed
                                    low-rank representation of the Hessian. I provided, all other parameters
                                    are ignored, if not, a new low-rank representation will be computed,
                                    using provided parameters.
    :param tol: The stopping criteria for the Lanczos algorithm.
                If `low_rank_representation` is provided, this parameter is ignored.
    :param max_iter: The maximum number of iterations for the Lanczos method.
                     If `low_rank_representation` is provided, this parameter is ignored.
    :param eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the model's
                                     device via cupy implementation.
                                     Make sure, that either your model is small enough or you use a
                                     small rank_estimate to fit your device's memory.
                                     If False, the eigen pair approximation is executed on the CPU by scipy wrapper to
                                     ARPACK.
    :return: Returns the solution vector x that satisfies the system Hx = b,
             where H is a low-rank approximation of the Hessian of the model's loss function.
    """

    b_device = b.device if hasattr(b, "device") else torch.device("cpu")

    if low_rank_representation is None:
        if b_device.type == "cuda" and not eigen_computation_on_gpu:
            raise ValueError(
                "Using 'eigen_computation_on_gpu=False' while 'b' is on a 'cuda' device is not supported. "
                "To address this, consider the following options:\n"
                " - Set eigen_computation_on_gpu=True if your model and data are small enough "
                "and if 'cupy' is available in your environment.\n"
                " - Move 'b' to the CPU with b.to('cpu').\n"
                " - Precompute a low rank representation and move it to the 'b' device using:\n"
                "     low_rank_representation = model_hessian_low_rank(model, training_data, ..., "
                "eigen_computation_on_gpu=False).to(b.device)"
            )

        low_rank_representation = model_hessian_low_rank(
            model,
            training_data,
            hessian_perturbation=hessian_perturbation,
            rank_estimate=rank_estimate,
            krylov_dimension=krylov_dimension,
            tol=tol,
            max_iter=max_iter,
            eigen_computation_on_gpu=eigen_computation_on_gpu,
        )
    else:
        if b_device.type != low_rank_representation.device.type:
            raise RuntimeError(
                f"The devices for 'b' and 'low_rank_representation' do not match.\n"
                f" - 'b' is on device: {b_device}\n"
                f" - 'low_rank_representation' is on device: {low_rank_representation.device}\n"
                f"\nTo resolve this, consider moving 'low_rank_representation' to '{b_device}' by using:\n"
                f"low_rank_representation = low_rank_representation.to(b.device)"
            )

        logger.info("Using provided low rank representation, ignoring other parameters")

    result = low_rank_representation.projections @ (
        torch.diag_embed(1.0 / low_rank_representation.eigen_vals)
        @ (low_rank_representation.projections.t() @ b.t())
    )
    return InverseHvpResult(
        x=result.t(),
        info={
            "eigenvalues": low_rank_representation.eigen_vals,
            "eigenvectors": low_rank_representation.projections,
        },
    )
