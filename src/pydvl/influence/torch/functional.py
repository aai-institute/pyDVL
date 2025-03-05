"""
This module provides methods for efficiently computing tensors related to first
and second order derivatives of torch models, using functionality from
[torch.func](https://pytorch.org/docs/stable/func.html).
To indicate higher-order functions, i.e. functions which return functions,
we use the naming convention `create_**_function`.

Among others, the module contains functionality for

* Sample, batch-wise and empirical loss functions:
    * [create_per_sample_loss_function][pydvl.influence.torch.functional.create_per_sample_loss_function]
    * [create_batch_loss_function][pydvl.influence.torch.functional.create_batch_loss_function]
    * [create_empirical_loss_function][pydvl.influence.torch.functional.create_empirical_loss_function]
* Per sample gradient and Jacobian product functions:
    * [create_per_sample_gradient_function][pydvl.influence.torch.functional.create_per_sample_gradient_function]
    * [create_per_sample_mixed_derivative_function][pydvl.influence.torch.functional.create_per_sample_mixed_derivative_function]
    * [create_matrix_jacobian_product_function][pydvl.influence.torch.functional.create_matrix_jacobian_product_function]
* Hessian, low rank approximation of Hessian and Hessian vector products:
    * [hvp][pydvl.influence.torch.functional.hvp]
    * [create_hvp_function][pydvl.influence.torch.functional.create_hvp_function]
    * [create_batch_hvp_function][pydvl.influence.torch.functional.create_batch_hvp_function]
    * [hessian][pydvl.influence.torch.functional.hessian]
    * [model_hessian_nystroem_approximation][pydvl.influence.torch.functional.model_hessian_nystroem_approximation]
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import torch
from scipy.sparse.linalg import ArpackNoConvergence
from torch._C import _LinAlgError
from torch.func import functional_call, grad, jvp, vjp
from torch.utils.data import DataLoader

from .util import (
    align_structure,
    align_with_model,
    flatten_dimensions,
    get_model_parameters,
    to_model_device,
)

if TYPE_CHECKING:
    from .base import TensorOperator

__all__ = [
    "create_hvp_function",
    "hessian",
    "create_batch_hvp_function",
    "create_per_sample_loss_function",
    "create_per_sample_gradient_function",
    "create_matrix_jacobian_product_function",
    "create_per_sample_mixed_derivative_function",
    "LowRankProductRepresentation",
    "randomized_nystroem_approximation",
    "model_hessian_nystroem_approximation",
    "create_batch_loss_function",
    "hvp",
    "operator_spectral_approximation",
    "operator_nystroem_approximation",
]


logger = logging.getLogger(__name__)


def hvp(
    func: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    params: Dict[str, torch.Tensor],
    vec: Dict[str, torch.Tensor],
    reverse_only: bool = True,
) -> Dict[str, torch.Tensor]:
    r"""
    Computes the Hessian-vector product (HVP) for a given function at the given
    parameters, i.e.

    \[\nabla_{\theta} \nabla_{\theta} f (\theta)\cdot v\]

    This function can operate in two modes, either reverse-mode autodiff only or both
    forward- and reverse-mode autodiff.

    Args:
        func: The scalar-valued function for which the HVP is computed.
        params: The parameters at which the HVP is computed.
        vec: The vector with which the Hessian is multiplied.
        reverse_only: Whether to use only reverse-mode autodiff
            (True, default) or both forward- and reverse-mode autodiff (False).

    Returns:
        The HVP of the function at the given parameters with the given vector.

    ??? Example

        ```pycon
        >>> def f(z): return torch.sum(z**2)
        >>> u = torch.ones(10, requires_grad=True)
        >>> v = torch.ones(10)
        >>> hvp_vec = hvp(f, u, v)
        >>> assert torch.allclose(hvp_vec, torch.full((10, ), 2.0))
        ```
    """

    output: Dict[str, torch.Tensor]

    if reverse_only:
        _, vjp_fn = vjp(grad(func), params)
        output = vjp_fn(vec)[0]
    else:
        output = jvp(grad(func), (params,), (vec,))[1]

    return output


def create_batch_hvp_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reverse_only: bool = True,
) -> Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]:
    r"""
    Creates a function to compute Hessian-vector product (HVP) for a given model and
    loss function, where the Hessian information is computed for a provided batch.

    This function takes a PyTorch model, a loss function,
    and an optional boolean parameter. It returns a callable
    that computes the Hessian-vector product for batches of input data
    and a given vector. The computation can be performed in reverse mode only,
    based on the `reverse_only` parameter.

    Args:
        model: The PyTorch model for which the Hessian-vector product is to be computed.
        loss: The loss function. It should take two
            torch.Tensor objects as input and return a torch.Tensor.
        reverse_only (bool, optional): If True, the Hessian-vector product is computed
            in reverse mode only.

    Returns:
        A function that takes three `torch.Tensor` objects - input data (`x`),
            target data (`y`), and a vector (`vec`),
            and returns the Hessian-vector product of the loss
            evaluated on `x`, `y` times `vec`.

    ??? Example
        ```python
        # Assume `model` is a PyTorch model and `loss_fn` is a loss function.
        b_hvp_function = batch_hvp(model, loss_fn)

        # `x_batch`, `y_batch` are batches of input and target data,
        # and `vec` is a vector.
        hvp_result = b_hvp_function(x_batch, y_batch, vec)
        ```
    """

    def b_hvp(
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        vec: torch.Tensor,
    ):
        return flatten_dimensions(
            hvp(
                lambda p: create_batch_loss_function(model, loss)(p, x, y),
                params,
                align_structure(params, vec),
                reverse_only=reverse_only,
            ).values()
        )

    return b_hvp


def create_empirical_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    r"""
    Creates a function to compute the empirical loss of a given model
    on a given dataset. If we denote the model parameters with \( \theta \),
    the resulting function approximates:

    \[
        f(\theta) = \frac{1}{N}\sum_{i=1}^N
        \operatorname{loss}(y_i, \operatorname{model}(\theta, x_i))
    \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$
    with model parameters $\theta$, where $N$ is the number of all elements provided
    by the data_loader.

    Args:
        model: The model for which the loss should be computed.
        loss: The loss function to be used.
        data_loader: The data loader for iterating over the dataset.

    Returns:
        A function that computes the empirical loss of the model on the dataset for
            given model parameters.

    """

    def empirical_loss(params: Dict[str, torch.Tensor]):
        total_loss = to_model_device(torch.zeros((), requires_grad=True), model)
        total_samples = to_model_device(torch.zeros(()), model)

        for x, y in iter(data_loader):
            output = functional_call(
                model,
                params,
                (to_model_device(x, model),),
            )
            loss_value = loss(output, to_model_device(y, model))
            total_loss = total_loss + loss_value * x.size(0)
            total_samples += x.size(0)

        return total_loss / total_samples

    return empirical_loss


def create_batch_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""
    Creates a function to compute the loss of a given model on a given batch of data,
    i.e. the function

    \[f(\theta, x, y) = \frac{1}{N} \sum_{i=1}^N
        \operatorname{loss}(\operatorname{model}(\theta, x_i), y_i)\]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$
    with model parameters $\theta$, where $N$ is the number of elements in the batch.
    Args:
        model: The model for which the loss should be computed.
        loss: The loss function to be used, which should be able to handle
            a batch dimension

    Returns:
        A function that computes the loss of the model on a batch for given
            model parameters. The model parameter input to the function must take
            the form of a dict conform to model.named_parameters(), i.e. the keys
            must be a subset of the parameters and the corresponding tensor shapes
            must align. For the data input, the first dimension has to be the batch
            dimension.
    """

    def batch_loss(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        outputs = functional_call(model, params, (to_model_device(x, model),))
        return loss(outputs, y)

    return batch_loss


def create_hvp_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    precompute_grad: bool = True,
    use_average: bool = True,
    reverse_only: bool = True,
    track_gradients: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that calculates the approximate Hessian-vector product
    for a given vector. If you want to compute the exact hessian,
    i.e., pulling all data into memory and compute a full gradient computation, use
    the function [hvp][pydvl.influence.torch.functional.hvp].

    Args:
        model: A PyTorch module representing the model whose loss function's
            Hessian is to be computed.
        loss: A callable that takes the model's output and target as input and
            returns the scalar loss.
        data_loader: A DataLoader instance that provides batches of data for
            calculating the Hessian-vector product. Each batch from the
            DataLoader is assumed to return a tuple where the first element is
            the model's input and the second element is the target output.
        precompute_grad: If `True`, the full data gradient is precomputed and
            kept in memory, which can speed up the hessian vector product
            computation. Set this to `False`, if you can't afford to keep the
            full computation graph in memory.
        use_average: If `True`, the returned function uses batch-wise
            computation via
            [a batch loss function][pydvl.influence.torch.functional.create_batch_loss_function]
            and averages the results.
            If `False`, the function uses backpropagation on the full
            [empirical loss function]
            [pydvl.influence.torch.functional.create_empirical_loss_function],
            which is more accurate than averaging the batch hessians, but
            probably has a way higher memory usage.
        reverse_only: Whether to use only reverse-mode autodiff or
            both forward- and reverse-mode autodiff. Ignored if
            `precompute_grad` is `True`.
        track_gradients: Whether to track gradients for the resulting tensor of
            the Hessian-vector products.

    Returns:
        A function that takes a single argument, a vector, and returns the
        product of the Hessian of the `loss` function with respect to the
        `model`'s parameters and the input vector.
    """

    if precompute_grad:
        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        if use_average:
            model_dtype = next(p.dtype for p in model.parameters() if p.requires_grad)
            total_grad_xy = torch.empty(0, dtype=model_dtype)
            total_points = 0
            grad_func = torch.func.grad(create_batch_loss_function(model, loss))
            for x, y in iter(data_loader):
                grad_xy = grad_func(
                    model_params, to_model_device(x, model), to_model_device(y, model)
                )
                grad_xy = flatten_dimensions(grad_xy.values())
                if total_grad_xy.nelement() == 0:
                    total_grad_xy = torch.zeros_like(grad_xy)
                total_grad_xy += grad_xy * len(x)
                total_points += len(x)
            total_grad_xy /= total_points
        else:
            total_grad_xy = torch.func.grad(
                create_empirical_loss_function(model, loss, data_loader)
            )(model_params)
            total_grad_xy = flatten_dimensions(total_grad_xy.values())

        def precomputed_grads_hvp_function(
            precomputed_grads: torch.Tensor, vec: torch.Tensor
        ) -> torch.Tensor:
            vec = to_model_device(vec, model)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)

            z = (precomputed_grads * torch.autograd.Variable(vec)).sum(dim=1)

            mvp = []
            for i in range(len(z)):
                mvp.append(
                    flatten_dimensions(
                        torch.autograd.grad(
                            z[i], list(model_params.values()), retain_graph=True
                        )
                    )
                )
            result = torch.stack([arr.contiguous().view(-1) for arr in mvp])

            if not track_gradients:
                result = result.detach()

            return result

        return partial(precomputed_grads_hvp_function, total_grad_xy)

    def hvp_function(vec: torch.Tensor) -> torch.Tensor:
        params = get_model_parameters(model, detach=not track_gradients)
        v = align_structure(params, vec)
        empirical_loss = create_empirical_loss_function(model, loss, data_loader)
        return flatten_dimensions(
            hvp(empirical_loss, params, v, reverse_only=reverse_only).values()
        )

    def avg_hvp_function(vec: torch.Tensor) -> torch.Tensor:
        n_batches = len(data_loader)
        avg_hessian = to_model_device(torch.zeros_like(vec), model)
        b_hvp = create_batch_hvp_function(model, loss, reverse_only)
        params = get_model_parameters(model, detach=not track_gradients)
        for t_x, t_y in iter(data_loader):
            t_x, t_y = to_model_device(t_x, model), to_model_device(t_y, model)
            avg_hessian += b_hvp(params, t_x, t_y, to_model_device(vec, model))

        return avg_hessian / float(n_batches)

    return avg_hvp_function if use_average else hvp_function


def hessian(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    use_hessian_avg: bool = True,
    track_gradients: bool = False,
    restrict_to: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Computes the Hessian matrix for a given model and loss function.

    Args:
        model: The PyTorch model for which the Hessian is computed.
        loss: A callable that computes the loss.
        data_loader: DataLoader providing batches of input data and corresponding
            ground truths.
        use_hessian_avg: Flag to indicate whether the average Hessian across
            mini-batches should be computed.
            If False, the empirical loss across the entire dataset is used.
        track_gradients: Whether to track gradients for the resulting tensor of
            the hessian vector products.
        restrict_to: The parameters to restrict the second order differentiation to,
            i.e. the corresponding sub-matrix of the Hessian. If None, the full Hessian
            is computed.

    Returns:
        A tensor representing the Hessian matrix. The shape of the tensor will be
            (n_parameters, n_parameters), where n_parameters is the number of trainable
            parameters in the model.
    """
    params = restrict_to

    if params is None:
        params = get_model_parameters(model, detach=not track_gradients)
    n_parameters = sum([p.numel() for p in params.values()])
    model_dtype = next((p.dtype for p in params.values()))

    flat_params = flatten_dimensions(params.values())

    if use_hessian_avg:
        n_samples = 0
        hessian_mat = to_model_device(
            torch.zeros((n_parameters, n_parameters), dtype=model_dtype), model
        )
        batch_loss = create_batch_loss_function(model, loss)

        def flat_input_batch_loss(
            p: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor
        ):
            return batch_loss(align_structure(params, p), t_x, t_y)

        for x, y in iter(data_loader):
            n_samples += x.shape[0]
            batch_hessian = torch.func.hessian(flat_input_batch_loss)(
                flat_params, to_model_device(x, model), to_model_device(y, model)
            )
            if not track_gradients and batch_hessian.requires_grad:
                batch_hessian = batch_hessian.detach()
            hessian_mat += x.shape[0] * batch_hessian

        hessian_mat /= n_samples
    else:

        def flat_input_empirical_loss(p: torch.Tensor):
            return create_empirical_loss_function(model, loss, data_loader)(
                align_with_model(p, model)
            )

        hessian_mat = torch.func.jacrev(torch.func.jacrev(flat_input_empirical_loss))(
            flat_params
        )

    return hessian_mat


def gauss_newton(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    restrict_to: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""
    Compute the Gauss-Newton matrix, i.e.

    $$ \sum_{i=1}^N \nabla_{\theta}\ell(m(x_i; \theta), y)
        \nabla_{\theta}\ell(m(x_i; \theta), y)^t,$$
    for a  loss function $\ell$ and a model $m$ with model parameters $\theta$.

    Args:
        model: The PyTorch model.
        loss: A callable that computes the loss.
        data_loader: A PyTorch DataLoader providing batches of input data and
            corresponding output data.
        restrict_to: The parameters to restrict the differentiation to,
            i.e. the corresponding sub-matrix of the Jacobian. If None, the full
            Jacobian is used.

    Returns:
        The Gauss-Newton matrix.
    """

    per_sample_grads = create_per_sample_gradient_function(model, loss)

    params = restrict_to
    if params is None:
        params = get_model_parameters(model)

    def generate_batch_matrices():
        for x, y in data_loader:
            grads = flatten_dimensions(
                per_sample_grads(params, x, y).values(), shape=(x.shape[0], -1)
            )
            batch_mat = grads.t() @ grads
            yield batch_mat.detach()

    n_points = 0
    tensors = generate_batch_matrices()
    result = next(tensors)

    for t in tensors:
        result += t
        n_points += t.shape[0]

    return result / n_points


def create_per_sample_loss_function(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""
    Generates a function to compute per-sample losses using PyTorch's vmap,
    i.e. the vector-valued function

    \[ f(\theta, x, y)  = (\operatorname{loss}(\operatorname{model}(\theta, x_1), y_1),
        \dots,
        \operatorname{loss}(\operatorname{model}(\theta, x_N), y_N)), \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with
    model parameters $\theta$, where $N$ is the number of elements in the batch.

    Args:
        model: The PyTorch model for which per-sample losses will be computed.
        loss: A callable that computes the loss.

    Returns:
        A callable that computes the loss for each sample in the batch,
            given a dictionary of model inputs, the model's predictions,
            and the true values. The callable will return a tensor where
            each entry corresponds to the loss of the corresponding sample.
    """

    def compute_loss(
        params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        outputs = functional_call(
            model, params, (to_model_device(x.unsqueeze(0), model),)
        )
        return loss(outputs, y.unsqueeze(0))

    vmap_loss: Callable[
        [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.vmap(compute_loss, in_dims=(None, 0, 0))
    return vmap_loss


def create_per_sample_gradient_function(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
]:
    r"""
    Generates a function to computes the per-sample gradient of the loss with respect to
    the model's parameters, i.e. the tensor-valued function

    \[ f(\theta, x, y) = (\nabla_{\theta}\operatorname{loss}
        (\operatorname{model}(\theta, x_1), y_1), \dots,
        \nabla_{\theta}\operatorname{loss}(\operatorname{model}(\theta, x_N), y_N) \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with
    model parameters $\theta$, where $N$ is the number of elements in the batch.

    Args:
        model: The PyTorch model for which per-sample gradients will be computed.
        loss: A callable that computes the loss.

    Returns:
        A callable that takes a dictionary of model parameters, the model's input,
            and the labels. It returns a dictionary with the same keys as the model's
            named parameters. Each entry in the returned dictionary corresponds to
            the gradient of the corresponding model parameter for each sample
            in the batch.

    """

    per_sample_grad: Callable[
        [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
    ] = torch.func.jacrev(create_per_sample_loss_function(model, loss))
    return per_sample_grad


def create_matrix_jacobian_product_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: torch.Tensor,
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""
    Generates a function to computes the matrix-Jacobian product (MJP) of the
    per-sample loss with respect to the model's parameters, i.e. the function

    \[ f(\theta, x, y) = g \, @ \, (\nabla_{\theta}\operatorname{loss}
        (\operatorname{model}(\theta, x_i), y_i))_i^T \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with
    model parameters $\theta$.

    Args:
        model: The PyTorch model for which the MJP will be computed.
        loss: A callable that computes the loss.
        g: Matrix for which the product with the Jacobian will be computed.
            The shape of this matrix should be consistent with the shape of
            the jacobian.

    Returns:
        A callable that takes a dictionary of model inputs, the model's input,
            and the labels. The callable returns the matrix-Jacobian product of the
            per-sample loss with respect to the model's parameters for the given
            matrix `g`.

    """

    def single_jvp(
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        _g: torch.Tensor,
    ):
        return torch.func.jvp(
            lambda p: create_per_sample_loss_function(model, loss)(p, x, y),
            (params,),
            (align_with_model(_g, model),),
        )[1]

    def full_jvp(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        return torch.func.vmap(single_jvp, in_dims=(None, None, None, 0))(
            params, x, y, g
        )

    return full_jvp


def create_per_sample_mixed_derivative_function(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
]:
    r"""
    Generates a function to computes the mixed derivatives, of the per-sample loss with
    respect to the model parameters and the input, i.e. the function

    \[ f(\theta, x, y) = \nabla_{\theta}\nabla_{x}\operatorname{loss}
        (\operatorname{model}(\theta, x), y) \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with
    model parameters $\theta$.

    Args:
        model: The PyTorch model for which the mixed derivatives are computed.
        loss: A callable that computes the loss.

    Returns:
        A callable that takes a dictionary of model inputs, the model's input,
            and the labels. The callable returns the mixed derivatives of the
            per-sample loss with respect to the model's parameters and input.

    """

    def compute_loss(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        outputs = functional_call(
            model, params, (to_model_device(x.unsqueeze(0), model),)
        )
        return loss(outputs, y.unsqueeze(0))

    per_samp_mix_derivative: Callable[
        [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
    ] = torch.vmap(
        torch.func.jacrev(torch.func.grad(compute_loss, argnums=1)),
        in_dims=(None, 0, 0),
    )
    return per_samp_mix_derivative


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

    @property
    def dtype(self) -> torch.dtype:
        return self.projections.dtype

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


def randomized_nystroem_approximation(
    mat_mat_prod: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    input_dim: int,
    rank: int,
    input_type: torch.dtype,
    shift_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mat_vec_device: torch.device = torch.device("cpu"),
) -> LowRankProductRepresentation:
    r"""
    Given a matrix vector product function (representing a symmetric positive definite
    matrix $A$ ), computes a random Nyström low rank approximation of
    $A$ in factored form, i.e.

    $$ A_{\text{nys}} = (A \Omega)(\Omega^T A \Omega)^{\dagger}(A \Omega)^T
    = U \Sigma U^T $$

    where $\Omega$ is a standard normal random matrix.

    Args:
        mat_mat_prod: A callable representing the matrix vector product
        input_dim: dimension of the input for the matrix vector product
        input_type: data_type of inputs
        rank: rank of the approximation
        shift_func: optional function for computing the stabilizing shift in the
            construction of the randomized nystroem approximation, defaults to

            $$ \sqrt{\operatorname{\text{input_dim}}} \cdot
                \varepsilon(\operatorname{\text{input_type}}) \cdot \|A\Omega\|_2,$$

            where $\varepsilon(\operatorname{\text{input_type}})$ is the value of the
            machine precision corresponding to the data type.
        mat_vec_device: device where the matrix vector product has to be executed

    Returns:
        object containing, $U$ and $\Sigma$
    """

    if shift_func is None:

        def shift_func(x: torch.Tensor):
            return (
                torch.sqrt(torch.as_tensor(input_dim))
                * torch.finfo(x.dtype).eps
                * torch.linalg.norm(x)
            )

    _mat_mat_prod: Callable[[torch.Tensor], torch.Tensor]

    if isinstance(mat_mat_prod, torch.Tensor):

        def _mat_mat_prod(x: torch.Tensor):
            return mat_mat_prod @ x

    else:
        _mat_mat_prod = mat_mat_prod

    random_sample_matrix = torch.randn(
        input_dim, rank, device=mat_vec_device, dtype=input_type
    )
    random_sample_matrix, _ = torch.linalg.qr(random_sample_matrix)

    sketch_mat = _mat_mat_prod(random_sample_matrix)

    shift = shift_func(sketch_mat)
    sketch_mat += shift * random_sample_matrix
    cholesky_mat = torch.matmul(random_sample_matrix.t(), sketch_mat)
    try:
        triangular_mat = torch.linalg.cholesky(cholesky_mat)
    except _LinAlgError as e:
        logger.warning(
            f"Encountered error in cholesky decomposition: {e}.\n "
            f"Increasing shift by smallest eigenvalue and re-compute"
        )
        eigen_vals, eigen_vectors = torch.linalg.eigh(cholesky_mat)
        shift += torch.abs(torch.min(eigen_vals))
        eigen_vals += shift
        triangular_mat = torch.linalg.cholesky(
            torch.mm(eigen_vectors, torch.mm(torch.diag(eigen_vals), eigen_vectors.T))
        )

    svd_input = torch.linalg.solve_triangular(
        triangular_mat.t(), sketch_mat, upper=True, left=False
    )
    left_singular_vecs, singular_vals, _ = torch.linalg.svd(
        svd_input, full_matrices=False
    )
    singular_vals = torch.clamp(singular_vals**2 - shift, min=0)

    return LowRankProductRepresentation(singular_vals, left_singular_vecs)


def model_hessian_nystroem_approximation(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    rank: int,
    shift_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LowRankProductRepresentation:
    r"""
    Given a model, loss and a data_loader, computes a random Nyström low rank approximation of
    the corresponding Hessian matrix in factored form, i.e.

    $$ H_{\text{nys}} = (H \Omega)(\Omega^T H \Omega)^{+}(H \Omega)^T
    = U \Sigma U^T $$

    Args:
        model: A PyTorch model instance. The Hessian will be calculated with respect to
            this model's parameters.
        loss : A callable that computes the loss.
        data_loader: A DataLoader instance that provides the model's training data.
            Used in calculating the Hessian-vector products.
        rank: rank of the approximation
        shift_func: optional function for computing the stabilizing shift in the
            construction of the randomized nystroem approximation, defaults to

            $$ \sqrt{\operatorname{\text{input_dim}}} \cdot
                \varepsilon(\operatorname{\text{input_type}}) \cdot \|A\Omega\|_2,$$

            where $\varepsilon(\operatorname{\text{input_type}})$ is the value of the
            machine precision corresponding to the data type.

    Returns:
        object containing, $U$ and $\Sigma$
    """

    model_hvp = create_hvp_function(
        model, loss, data_loader, precompute_grad=False, use_average=True
    )
    device = next((p.device for p in model.parameters()))
    dtype = next((p.dtype for p in model.parameters()))
    in_dim = sum((p.numel() for p in model.parameters() if p.requires_grad))

    def model_hessian_mat_mat_prod(x: torch.Tensor):
        return torch.func.vmap(model_hvp, in_dims=1, randomness="same")(x).t()

    return randomized_nystroem_approximation(
        model_hessian_mat_mat_prod,
        in_dim,
        rank,
        dtype,
        shift_func=shift_func,
        mat_vec_device=device,
    )


def operator_nystroem_approximation(
    operator: "TensorOperator",
    rank: int,
    shift_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> LowRankProductRepresentation:
    r"""
    Given an operator (representing a symmetric positive definite
    matrix $A$ ), computes a random Nyström low rank approximation of
    $A$ in factored form, i.e.

    $$ A_{\text{nys}} = (A \Omega)(\Omega^T A \Omega)^{\dagger}(A \Omega)^T
    = U \Sigma U^T $$

    where $\Omega$ is a standard normal random matrix.

    Args:
        operator: the operator to approximate
        rank: rank of the approximation
        shift_func: optional function for computing the stabilizing shift in the
            construction of the randomized nystroem approximation, defaults to

            $$ \sqrt{\operatorname{\text{input_dim}}} \cdot
                \varepsilon(\operatorname{\text{input_type}}) \cdot \|A\Omega\|_2,$$

            where $\varepsilon(\operatorname{\text{input_type}})$ is the value of the
            machine precision corresponding to the data type.

    Returns:
        object containing, $U$ and $\Sigma$
    """

    def mat_mat_prod(x: torch.Tensor):
        return operator.apply(x.t()).t()

    return randomized_nystroem_approximation(
        mat_mat_prod,
        operator.input_size,
        rank,
        operator.dtype,
        shift_func=shift_func,
        mat_vec_device=operator.device,
    )


def operator_spectral_approximation(
    operator: "TensorOperator",
    rank: int = 10,
    krylov_dimension: Optional[int] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    eigen_computation_on_gpu: bool = False,
) -> "LowRankProductRepresentation":
    r"""
    Calculates a low-rank approximation of an operator $H$ using the implicitly
    restarted Lanczos algorithm, i.e.:

    \[ H_{\text{approx}} = V D V^T\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank`
    eigenvalues of the Hessian and \(V\) contains the corresponding eigenvectors.

    Args:
        operator: The operator to approximate.
        rank: The number of eigenvalues and corresponding eigenvectors
            to compute. Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos
            method. If not provided, it defaults to
            \( \min(\text{model.n_parameters},
                \max(2 \times \text{rank_estimate} + 1, 20)) \).
        tol: The stopping criteria for the Lanczos algorithm, which stops when
            the difference in the approximated eigenvalue is less than `tol`.
            Defaults to 1e-6.
        max_iter: The maximum number of iterations for the Lanczos method. If
            not provided, it defaults to \( 10 \cdot \text{model.n_parameters}\).
        eigen_computation_on_gpu: If True, tries to execute the eigen pair
            approximation on the provided device via [cupy](https://cupy.dev/)
            implementation. Ensure that either your model is small enough, or you
            use a small rank_estimate to fit your device's memory. If False, the
            eigen pair approximation is executed on the CPU with scipy's wrapper to
            ARPACK.

    Returns:
        [LowRankProductRepresentation]
            [pydvl.influence.torch.functional.LowRankProductRepresentation]
            instance that contains the top (up until rank_estimate) eigenvalues
            and corresponding eigenvectors of the Hessian.
    """

    if operator.input_size == 1:
        # in the trivial case, return early
        eigen_vec = torch.ones((1, 1), dtype=operator.dtype, device=operator.device)
        eigen_val = operator.apply(eigen_vec).squeeze()
        return LowRankProductRepresentation(eigen_val, eigen_vec)

    torch_dtype = operator.dtype

    if eigen_computation_on_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
            from torch.utils.dlpack import from_dlpack, to_dlpack
        except ImportError as e:
            raise ImportError(
                "Missing cupy, check the installation instructions "
                "at https://docs.cupy.dev/en/stable/install.html "
                "or set eigen_computation_on_gpu "
                f"to False: {e}"
            )

        def to_torch_conversion_function(x: cp.NDArray) -> torch.Tensor:
            return from_dlpack(x.toDlpack()).to(torch_dtype)

        def mv(x):
            x = to_torch_conversion_function(x)
            y = operator.apply(x)
            return cp.from_dlpack(to_dlpack(y))

    else:
        from scipy.sparse.linalg import LinearOperator, eigsh

        def mv(x):
            x_torch = torch.as_tensor(x, device=operator.device, dtype=torch_dtype)
            y = operator.apply(x_torch).detach().cpu().numpy()
            return y

        to_torch_conversion_function = partial(torch.as_tensor, dtype=torch_dtype)

    try:
        matrix_shape = (operator.input_size, operator.input_size)
        eigen_vals, eigen_vecs = eigsh(
            LinearOperator(matrix_shape, matvec=mv),
            k=min(rank, operator.input_size - 1),
            maxiter=max_iter,
            tol=tol,
            ncv=krylov_dimension,
            return_eigenvectors=True,
        )

    except ArpackNoConvergence as e:
        warnings.warn(
            f"ARPACK did not converge for parameters {max_iter=}, {tol=}, "
            f"{krylov_dimension=}, {rank=}. \n "
            f"Returning the best approximation found so far. "
            f"Use those with care or modify parameters.\n Original error: {e}"
        )

        eigen_vals, eigen_vecs = e.eigenvalues, e.eigenvectors

    eigen_vals = to_torch_conversion_function(eigen_vals)
    eigen_vecs = to_torch_conversion_function(eigen_vecs)

    return LowRankProductRepresentation(eigen_vals, eigen_vecs)
