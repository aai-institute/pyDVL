"""
Contains methods for differentiating  a pyTorch model. Most of the methods focus
on ways to calculate matrix vector products. Moreover, it contains several
methods to invert the Hessian vector product. These are used to calculate the
influence of a training point on the model.

## References

[^1]: <a name="koh_liang_2017"></a>Koh, P.W., Liang, P., 2017.
    [Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html).
    In: Proceedings of the 34th International Conference on Machine Learning, pp. 1885–1894. PMLR.
[^2]: <a name="agarwal_secondorder_2017"></a>Agarwal, N., Bullins, B., Hazan, E., 2017.
    [Second-Order Stochastic Optimization for Machine Learning in Linear Time](https://www.jmlr.org/papers/v18/16-491.html).
    In: Journal of Machine Learning Research, Vol. 18, pp. 1–40. JMLR.
"""
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generator, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object import PMatEKFAC
from numpy import isin
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
    "model_hessian_low_rank",
]

logger = logging.getLogger(__name__)


class TorchTwiceDifferentiable(TwiceDifferentiable[torch.Tensor]):
    r"""
    Wraps a [torch.nn.Module][torch.nn.Module]
    and a loss function and provides methods to compute gradients and
    second derivative of the loss wrt. the model parameters

    Args:
        model: A (differentiable) function.
        loss: A differentiable scalar loss \( L(\hat{y}, y) \),
            mapping a prediction and a target to a real value.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):

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
        """
        Returns:
            All model parameters that require differentiating.
        """

        return [param for param in self.model.parameters() if param.requires_grad]

    @property
    def num_params(self) -> int:
        """
        Get the number of parameters of model f.

        Returns:
            int: Number of parameters.
        """
        return sum([p.numel() for p in self.parameters])

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        r"""
        Calculates gradient of model parameters with respect to the model parameters.

        Args:
            x: A matrix [NxD] representing the features \( x_i \).
            y: A matrix [NxK] representing the target values \( y_i \).
            create_graph (bool): If True, the resulting gradient tensor can be used for further differentiation.

        Returns:
            An array [P] with the gradients of the model.
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
        r"""
        Calculates the explicit hessian of model parameters given data \(x\) and \(y\).

        Args:
            x: A matrix [NxD] representing the features \(x_i\).
            y: A matrix [NxK] representing the target values \(y_i\).

        Returns:
            A tensor representing the hessian of the loss with respect to the model parameters.
        """

        def model_func(param):
            outputs = torch.func.functional_call(
                self.model,
                align_structure(
                    {k: p for k, p in self.model.named_parameters() if p.requires_grad},
                    param,
                ),
                (x.to(self.device),),
            )
            return self.loss(outputs, y.to(self.device))

        params = flatten_tensors_to_vector(
            p.detach() for p in self.model.parameters() if p.requires_grad
        )
        return torch.func.hessian(model_func)(params)  # type: ignore

    @staticmethod
    def mvp(
        grad_xy: torch.Tensor,
        v: torch.Tensor,
        backprop_on: Union[torch.Tensor, Sequence[torch.Tensor]],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        r"""
        Calculates the second-order derivative of the model along directions v.
        This second-order derivative can be selected through the `backprop_on` argument.

        Args:
            grad_xy: An array [P] holding the gradients of the model parameters with respect to input
                \(x\) and labels \(y\), where P is the number of parameters of the model.
                It is typically obtained through `self.grad`.
            v: An array ([DxP] or even one-dimensional [D]) which multiplies the matrix,
                where D is the number of directions.
            progress: If True, progress will be printed.
            backprop_on: Tensor used in the second backpropagation
                (the first one is defined via grad_xy).

        Returns:
            A matrix representing the implicit matrix-vector product of the model along the given directions.
                The output shape is [DxM], with M being the number of elements of `backprop_on`.
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
    r"""
    Representation of a low rank product of the form \(H = V D V^T\),
    where D is a diagonal matrix and V is orthogonal.

    Args:
        eigen_vals: Diagonal of D.
        projections: The matrix V.
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
    torch_dtype: Optional[torch.dtype] = None,
) -> LowRankProductRepresentation:
    r"""
    Calculates a low-rank approximation of the Hessian matrix of a scalar-valued
    function using the implicitly restarted Lanczos algorithm, i.e.:

    \[ H_{\text{approx}} = V D V^T\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank_estimate` eigenvalues of the Hessian
    and \(V\) contains the corresponding eigenvectors.

    Args:
        hessian_vp: A function that takes a vector and returns the product of
            the Hessian of the loss function.
        matrix_shape: The shape of the matrix, represented by the hessian vector
            product.
        hessian_perturbation: Regularization parameter added to the
            Hessian-vector product for numerical stability.
        rank_estimate: The number of eigenvalues and corresponding eigenvectors
            to compute. Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos
            method. If not provided, it defaults to
            \( \min(\text{model.num_parameters}, \max(2 \times \text{rank_estimate} + 1, 20)) \).
        tol: The stopping criteria for the Lanczos algorithm, which stops when
            the difference in the approximated eigenvalue is less than `tol`.
            Defaults to 1e-6.
        max_iter: The maximum number of iterations for the Lanczos method. If
            not provided, it defaults to \( 10 \cdot \text{model.num_parameters}\).
        device: The device to use for executing the hessian vector product.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair
            approximation on the provided device via [cupy](https://cupy.dev/)
            implementation. Ensure that either your model is small enough, or you
            use a small rank_estimate to fit your device's memory. If False, the
            eigen pair approximation is executed on the CPU with scipy's wrapper to
            ARPACK.
        torch_dtype: If not provided, the current torch default dtype is used for
            conversion to torch.

    Returns:
        A [LowRankProductRepresentation][pydvl.influence.torch.torch_differentiable.LowRankProductRepresentation]
            instance that contains the top (up until rank_estimate) eigenvalues
            and corresponding eigenvectors of the Hessian.
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

        def to_torch_conversion_function(x: cp.NDArray) -> torch.Tensor:
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
    r"""
    Calculates a low-rank approximation of the Hessian matrix of the model's loss function using the implicitly
    restarted Lanczos algorithm, i.e.

    \[ H_{\text{approx}} = V D V^T\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank_estimate` eigenvalues of the Hessian
    and \(V\) contains the corresponding eigenvectors.


    Args:
        model: A PyTorch model instance that is twice differentiable, wrapped into `TorchTwiceDifferential`.
            The Hessian will be calculated with respect to this model's parameters.
        training_data: A DataLoader instance that provides the model's training data.
            Used in calculating the Hessian-vector products.
        hessian_perturbation: Optional regularization parameter added to the Hessian-vector product
            for numerical stability.
        rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
            Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
            If not provided, it defaults to min(model.num_parameters, max(2*rank_estimate + 1, 20)).
        tol: The stopping criteria for the Lanczos algorithm, which stops when the difference
            in the approximated eigenvalue is less than `tol`. Defaults to 1e-6.
        max_iter: The maximum number of iterations for the Lanczos method. If not provided, it defaults to
            10*model.num_parameters.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the provided
            device via cupy implementation.
            Make sure, that either your model is small enough or you use a
            small rank_estimate to fit your device's memory.
            If False, the eigen pair approximation is executed on the CPU by scipy wrapper to
            ARPACK.

    Returns:
        A [LowRankProductRepresentation][pydvl.influence.torch.torch_differentiable.LowRankProductRepresentation]
            instance that contains the top (up until rank_estimate) eigenvalues
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


def get_layer_kfac_hooks(m_name, module, forward_x, grad_y):
    if isinstance(module, nn.Linear):
        with_bias = module.bias is not None

        def input_hook(m, x, y):
            x = x[0]
            if with_bias:
                x = torch.cat(
                    (x, torch.ones((x.shape[0], 1), device=module.weight.device)), dim=1
                )
            forward_x[m_name] += torch.mm(x.t(), x)

        def grad_hook(m, m_grad, m_out):
            m_out = m_out[0]
            grad_y[m_name] += torch.mm(m_out.t(), m_out)

    elif isinstance(module, nn.Conv2d):
        pass

    else:
        supported_layers = ["Linear", "Conv2d"]
        raise NotImplementedError(
            f"Only {supported_layers} layers are supported, but found module {module} requiring grad."
        )
    return input_hook, grad_hook


def init_layer_kfac_blocks(module):
    if isinstance(module, nn.Linear):
        with_bias = module.bias is not None
        sG = module.out_features
        sA = module.in_features + int(with_bias)
        forward_x_layer = torch.zeros((sA, sA), device=module.weight.device)
        grad_y_layer = torch.zeros((sG, sG), device=module.weight.device)
    elif isinstance(module, nn.Conv2d):
        pass
    else:
        raise NotImplementedError(
            f"Only Linear and Conv2d layers are supported, but found module {module} requiring grad."
        )
    return forward_x_layer, grad_y_layer


def get_kfac_blocks(
    empirical_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    model: TorchTwiceDifferentiable,
    data_loader: DataLoader,
):
    forward_x = {}
    grad_y = {}
    hooks = []
    data_len = len(data_loader.sampler)

    for m_name, module in model.model.named_modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            layer_requires_grad = [param.requires_grad for param in module.parameters()]
            if any(layer_requires_grad):
                forward_x[m_name], grad_y[m_name] = init_layer_kfac_blocks(module)
                layer_input_hook, layer_grad_hook = get_layer_kfac_hooks(
                    m_name, module, forward_x, grad_y
                )
                hooks.append(module.register_forward_hook(layer_input_hook))
                hooks.append(module.register_full_backward_hook(layer_grad_hook))

    for x, _ in data_loader:
        pred_y = model.model(x)
        loss = empirical_loss_fn(pred_y)
        loss.backward()

    for key in forward_x.keys():
        forward_x[key] /= data_len
        grad_y[key] /= data_len

    for hook in hooks:
        hook.remove()

    return forward_x, grad_y


def init_layer_diag(module):
    if isinstance(module, nn.Linear):
        with_bias = module.bias is not None
        sG = module.out_features
        sA = module.in_features + int(with_bias)
        layer_diag = torch.zeros((sA * sG), device=module.weight.device)
    elif isinstance(module, nn.Conv2d):
        pass
    else:
        raise NotImplementedError(
            f"Only Linear and Conv2d layers are supported, but found module {module} requiring grad."
        )
    return layer_diag


def get_layer_diag_hooks(m_name, module, last_x_kfe, diags, evecs):
    if isinstance(module, nn.Linear):
        with_bias = module.bias is not None

        def input_hook(m, x, y):
            x = x[0]
            if with_bias:
                x = torch.cat(
                    (x, torch.ones((x.shape[0], 1), device=module.weight.device)), dim=1
                )
            last_x_kfe[m_name] = torch.mm(x, evecs[m_name][0])

        def grad_hook(m, m_grad, m_out):
            m_out = m_out[0]
            gy_kfe = torch.mm(m_out, evecs[m_name][1])
            diags[m_name] += torch.mm(gy_kfe.t() ** 2, last_x_kfe[m_name] ** 2).view(-1)

    elif isinstance(module, nn.Conv2d):
        pass

    else:
        supported_layers = ["Linear", "Conv2d"]
        raise NotImplementedError(
            f"Only {supported_layers} layers are supported, but found module {module} requiring grad."
        )
    return input_hook, grad_hook


def get_updated_diag(
    empirical_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    model: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    evecs,
):
    diags = {}
    last_x_kfe = {}
    hooks = []
    data_len = len(data_loader.sampler)

    for m_name, module in model.model.named_modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            layer_requires_grad = [param.requires_grad for param in module.parameters()]
            if any(layer_requires_grad):
                diags[m_name] = init_layer_diag(module)
                input_hook, grad_hook = get_layer_diag_hooks(
                    m_name, module, last_x_kfe, diags, evecs
                )
                hooks.append(module.register_forward_hook(input_hook))
                hooks.append(module.register_full_backward_hook(grad_hook))
    for x, _ in data_loader:
        pred_y = model.model(x)
        loss = empirical_loss_fn(pred_y)
        loss.backward()

    for key in diags.keys():
        diags[key] /= data_len

    for hook in hooks:
        hook.remove()

    return diags


def get_ekfac_representation(loss_fn, model, data_loader, update_diag=True):
    """
    Get KFAC representation of the loss function for the given model and data.

    Args:
        loss_fn: Loss function.
        model: Model.
        data_loader: Data loader.

    Returns:
        TODO
    """
    forward_x, grad_y = get_kfac_blocks(loss_fn, model, data_loader)
    evecs = {}
    diags = {}
    for key in forward_x.keys():
        evals_a, evecs_a = torch.linalg.eigh(forward_x[key])
        evals_g, evecs_g = torch.linalg.eigh(grad_y[key])
        evecs[key] = (evecs_a, evecs_g)
        diags[key] = torch.kron(evals_g.view(-1, 1), evals_a.view(-1, 1))
    if update_diag:
        diags = get_updated_diag(loss_fn, model, data_loader, evecs)
    return evecs, diags


def solve_ekfac_ihvp(evecs, diags, b, hessian_perturbation=0.0):
    x = b.clone()
    start_idx = 0
    for layer_id in diags.keys():
        evecs_a, evecs_g = evecs[layer_id]
        diag = diags[layer_id]
        end_idx = start_idx + diag.shape[0]
        b_trial = b[:, start_idx : end_idx - evecs_g.shape[0]].reshape(
            b.shape[0], evecs_g.shape[0], -1
        )
        bias_v = b[:, end_idx - evecs_g.shape[0] : end_idx]
        b_trial = torch.cat([b_trial, bias_v.unsqueeze(2)], dim=2)
        v_kfe = torch.einsum(
            "bij,jk->bik", torch.einsum("ij,bjk->bik", evecs_g.t(), b_trial), evecs_a
        )
        inv_diag = 1 / (
            diags[layer_id].reshape(*v_kfe.shape[1:]) + hessian_perturbation
        )
        inv_kfe = torch.einsum("bij,ij->bij", v_kfe, inv_diag)
        inv = torch.einsum(
            "bij,jk->bik", torch.einsum("ij,bjk->bik", evecs_g, inv_kfe), evecs_a.t()
        )
        x[:, start_idx:end_idx] = torch.cat(
            [inv[:, :, :-1].reshape(b.shape[0], -1), inv[:, :, -1]], dim=1
        )
        start_idx = end_idx
    x.detach_()
    return x


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
        return torch.cat(a, **kwargs)  # type: ignore

    @staticmethod
    def stack(a: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        """Stacks a sequence of tensors into a single torch tensor"""
        return torch.stack(a, **kwargs)  # type: ignore

    @staticmethod
    def unsqueeze(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Add a singleton dimension at a specified position in a tensor.

        Args:
            x: A PyTorch tensor.
            dim: The position at which to add the singleton dimension. Zero-based indexing.

        Returns:
            A new tensor with an additional singleton dimension.
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
    r"""
    Given a model and training data, it finds x such that \(Hx = b\), with \(H\) being the model hessian.

    Args:
        model: A model wrapped in the TwiceDifferentiable interface.
        training_data: A DataLoader containing the training data.
        b: A vector or matrix, the right hand side of the equation \(Hx = b\).
        hessian_perturbation: Regularization of the hessian.

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult],
            having an array that solves the inverse problem, i.e. it returns \(x\) such that \(Hx = b\),
            and a dictionary containing information about the solution.
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
    try:
        x = torch.linalg.solve(matrix, b.T).T
    except torch.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Direct inversion failed, possibly due to the Hessian being singular. "
            f"Consider increasing the parameter 'hessian_perturbation' (currently: {hessian_perturbation}). \n{e}"
        )
    return InverseHvpResult(x=x, info=info)


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
    r"""
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the Hessian Vector Product. More precisely, it finds x such that \(Hx =
    b\), with \(H\) being the model hessian. For more info, see
    [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

    Args:
        model: A model wrapped in the TwiceDifferentiable interface.
        training_data: A DataLoader containing the training data.
        b: A vector or matrix, the right hand side of the equation \(Hx = b\).
        hessian_perturbation: Regularization of the hessian.
        x0: Initial guess for hvp. If None, defaults to b.
        rtol: Maximum relative tolerance of result.
        atol: Absolute tolerance of result.
        maxiter: Maximum number of iterations. If None, defaults to 10*len(b).
        progress: If True, display progress bars.

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult],
            having a matrix of shape [NxP] with each line being a solution of \(Ax=b\),
            and a dictionary containing information about the convergence of CG,
            one entry for each line of the matrix.
    """
    if len(training_data) == 0:
        raise ValueError("Training dataloader must not be empty.")

    total_grad_xy = torch.empty(0)
    total_points = 0

    for x, y in maybe_progress(training_data, progress, desc="Batch Train Gradients"):
        grad_xy = model.grad(x, y, create_graph=True)
        if total_grad_xy.nelement() == 0:
            total_grad_xy = torch.zeros_like(grad_xy)
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
    r"""
    Conjugate gradient solver for the Hessian vector product.

    Args:
        hvp: A callable Hvp, operating with tensors of size N.
        b: A vector or matrix, the right hand side of the equation \(Hx = b\).
        x0: Initial guess for hvp.
        rtol: Maximum relative tolerance of result.
        atol: Absolute tolerance of result.
        maxiter: Maximum number of iterations. If None, defaults to 10*len(b).

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult],
            with a vector x, solution of \(Ax=b\), and a dictionary containing
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
    approximate the inverse Hessian. More precisely, it finds x s.t. \(Hx = b\),
    with \(H\) being the model's second derivative wrt. the parameters.
    This is done with the update

    \[H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},\]

    where \(I\) is the identity matrix, \(d\) is a dampening term and \(s\) a scaling
    factor that are applied to help convergence. For details, see
    (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup> and the original paper
    (Agarwal et. al.)<sup><a href="#agarwal_secondorder_2017">2</a></sup>.

    Args:
        model: A model wrapped in the TwiceDifferentiable interface.
        training_data: A DataLoader containing the training data.
        b: A vector or matrix, the right hand side of the equation \(Hx = b\).
        hessian_perturbation: Regularization of the hessian.
        maxiter: Maximum number of iterations.
        dampen: Dampening factor, defaults to 0 for no dampening.
        scale: Scaling factor, defaults to 10.
        h0: Initial guess for hvp.
        rtol: tolerance to use for early stopping
        progress: If True, display progress bars.

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult], with a matrix of shape [NxP] with each line being a solution of \(Ax=b\),
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

        Args:
            h: An estimate of the hessian inverse.
            reg_hvp: Regularised hessian vector product.

        Returns:
            The next estimate of the hessian inverse.
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


@InversionRegistry.register(TorchTwiceDifferentiable, InversionMethod.Ekfac)
def solve_ekfac(
    model: TorchTwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    hessian_perturbation: float = 0.0,
    update_diag: bool = True,
    local_implemetation: bool = True,
    on_layers: Optional[Sequence[str]] = None,
) -> InverseHvpResult:

    if local_implemetation:

        def batch_loss_fn(model_output: torch.Tensor) -> torch.Tensor:
            probs_ = torch.softmax(model_output, dim=1)
            log_probs_ = torch.log(probs_)
            log_probs_ = torch.where(
                torch.isfinite(log_probs_), log_probs_, torch.zeros_like(log_probs_)
            )
            return torch.sum(log_probs_ * probs_.detach() ** 0.5)

        evecs, diags = get_ekfac_representation(
            batch_loss_fn, model, training_data, update_diag=update_diag
        )
        x = solve_ekfac_ihvp(evecs, diags, b, hessian_perturbation=hessian_perturbation)
        info = {"hessian_repr": (evecs, diags), "x": x}
    else:

        def batch_loss_fn(*d) -> torch.Tensor:
            probs_ = torch.softmax(model.model(d[0].to(model.device)), dim=1)
            log_probs_ = torch.log(probs_)
            log_probs_ = torch.where(
                torch.isfinite(log_probs_), log_probs_, torch.zeros_like(log_probs_)
            )
            return torch.sum(log_probs_ * probs_.detach() ** 0.5, axis=1)

        active_layers = LayerCollection()
        for name, module in model.model.named_modules():
            if on_layers is None:
                if (
                    len(list(module.children())) == 0
                    and len(list(module.parameters())) > 0
                ):
                    layer_requires_grad = [
                        param.requires_grad for param in module.parameters()
                    ]
                    if any(layer_requires_grad):
                        active_layers.add_layer(
                            "%s.%s" % (name, str(module)),
                            LayerCollection._module_to_layer(module),
                        )
            elif name in on_layers:
                active_layers.add_layer(
                    "%s.%s" % (name, str(module)),
                    LayerCollection._module_to_layer(module),
                )

        hessian_repr: PMatEKFAC = FIM(
            model.model,
            training_data,
            representation=PMatEKFAC,
            n_output=1,
            function=batch_loss_fn,
            # HACK: in order to pass the correct function,
            # we need to set this to "regression", even though we are doing classification.
            # Maybe open issue in nngeometry
            variant="regression",
            layer_collection=active_layers,
        )

        if update_diag:
            hessian_repr.update_diag(training_data)
        x = b.clone()
        try:
            inverse_hessian_repr = hessian_repr.inverse(regul=hessian_perturbation)
        except Exception as e:
            raise RuntimeError(
                "Exception encountered, possibly due to the Hessian being singular. "
                f"Consider increasing the parameter 'hessian_perturbation' (currently: {hessian_perturbation}). \n{e}"
            )
        _, diags = inverse_hessian_repr.data
        kfe_layers = inverse_hessian_repr.get_KFE()
        for layer_id in inverse_hessian_repr.generator.layer_collection.layers.keys():
            diag = diags[layer_id]
            start = inverse_hessian_repr.generator.layer_collection.p_pos[layer_id]
            sAG = diag.numel()
            kfe_layer = kfe_layers[layer_id]
            layer_block = torch.mm(
                kfe_layer, torch.mm(torch.diag(diag.view(-1)), kfe_layer.t())
            ).to(model.device)
            x[:, start : start + sAG] = b[:, start : start + sAG] @ layer_block
        info = {"hessian_repr": hessian_repr.data, "x": x}
    return InverseHvpResult(x=x, info=info)


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
    r"""
    Solves the linear system Hx = b, where H is the Hessian of the model's loss function and b is the given
    right-hand side vector.
    It employs the [implicitly restarted Arnoldi method](https://en.wikipedia.org/wiki/Arnoldi_iteration) for
    computing a partial eigen decomposition, which is used fo the inversion i.e.

    \[x = V D^{-1} V^T b\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank_estimate` eigenvalues of the Hessian
    and \(V\) contains the corresponding eigenvectors.

    Args:
        model: A PyTorch model instance that is twice differentiable, wrapped into
            [TorchTwiceDifferential][pydvl.influence.torch.torch_differentiable.TorchTwiceDifferentiable].
            The Hessian will be calculated with respect to this model's parameters.
        training_data: A DataLoader instance that provides the model's training data.
            Used in calculating the Hessian-vector products.
        b: The right-hand side vector in the system Hx = b.
        hessian_perturbation: Optional regularization parameter added to the Hessian-vector
            product for numerical stability.
        rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
            Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
            Defaults to min(model's number of parameters, max(2 times rank_estimate + 1, 20)).
        low_rank_representation: An instance of
            [LowRankProductRepresentation][pydvl.influence.torch.torch_differentiable.LowRankProductRepresentation]
            containing a previously computed low-rank representation of the Hessian. If provided, all other parameters
            are ignored; otherwise, a new low-rank representation is computed
            using provided parameters.
        tol: The stopping criteria for the Lanczos algorithm.
            Ignored if `low_rank_representation` is provided.
        max_iter: The maximum number of iterations for the Lanczos method.
            Ignored if `low_rank_representation` is provided.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the model's device
            via a cupy implementation. Ensure the model size or rank_estimate is appropriate for device memory.
            If False, the eigen pair approximation is executed on the CPU by the scipy wrapper to ARPACK.

    Returns:
        Instance of [InverseHvpResult][pydvl.influence.torch.torch_differentiable.InverseHvpResult],
            having the solution vector x that satisfies the system \(Ax = b\),
            where \(A\) is a low-rank approximation of the Hessian \(H\) of the model's loss function, and an instance
            of [LowRankProductRepresentation][pydvl.influence.torch.torch_differentiable.LowRankProductRepresentation],
            which represents the approximation of H.
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
