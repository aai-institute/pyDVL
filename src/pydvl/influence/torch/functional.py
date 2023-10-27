import functools
from typing import Callable, Dict, Generator

import torch
from torch.func import functional_call, grad, jvp, vjp
from torch.utils.data import DataLoader

from .util import (
    TorchTensorContainerType,
    align_structure,
    align_with_model,
    flatten_dimensions,
    to_model_device,
)

__all__ = [
    "get_hvp_function",
    "get_hessian",
    "per_sample_loss",
    "per_sample_gradient",
    "matrix_jacobian_product",
    "per_sample_mixed_derivative",
]


def hvp(
    func: Callable[[TorchTensorContainerType], torch.Tensor],
    params: TorchTensorContainerType,
    vec: TorchTensorContainerType,
    reverse_only: bool = True,
) -> TorchTensorContainerType:
    r"""
    Computes the Hessian-vector product (HVP) for a given function at given parameters, i.e.

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

    Example:
    ```python
    >>> def f(z): return torch.sum(z**2)
    >>> u = torch.ones(10, requires_grad=True)
    >>> v = torch.ones(10)
    >>> hvp_vec = hvp(f, u, v)
    >>> assert torch.allclose(hvp_vec, torch.full((10, ), 2.0))
    ```
    """

    output: TorchTensorContainerType

    if reverse_only:
        _, vjp_fn = vjp(grad(func), params)
        output = vjp_fn(vec)[0]
    else:
        output = jvp(grad(func), (params,), (vec,))[1]

    return output


def get_batch_hvp(
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reverse_only: bool = True,
        detach: bool = True
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Creates a function to compute the batch-wise Hessian-vector product (HVP) for a given model and loss function.

    This function takes a PyTorch model, a loss function, and optional boolean parameters. It returns a callable
    that computes the Hessian-vector product for batches of input data and a given vector. The computation can be
    performed in reverse mode only, based on the `reverse_only` parameter. Additionally, the function allows
    detaching of the model's parameters from the current computation graph based on the `detach` parameter.

    Args:
        model: The PyTorch model for which the Hessian-vector product is to be computed.
        loss: The loss function. It should take two
            torch.Tensor objects as input and return a torch.Tensor.
        reverse_only (bool, optional): If True, the Hessian-vector product is computed in reverse mode only. Defaults
            to True.
        detach (bool, optional): If True, the model's parameters are detached from the current computation graph
            before computation. This can save memory during gradient calculations. Defaults to True.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: A function that takes three torch.Tensor
        objects - input data (x), target data (y), and a vector (vec) - and returns the Hessian-vector product as a
        torch.Tensor.

    Example Usage:
        # Assume `model` is a PyTorch model and `loss_fn` is a loss function.
        b_hvp_function = batch_hvp(model, loss_fn)

        # `x_batch`, `y_batch` are batches of input and target data, and `vec` is a vector.
        hvp_result = b_hvp_function(x_batch, y_batch, vec)

    Note:
        The returned function internally manages model parameters based on the `detach` argument. When `detach` is
        True, it detaches the parameters from the computation graph which can be beneficial for reducing memory usage
        during gradient calculations.
    """

    def b_hvp(x: torch.Tensor, y: torch.Tensor, vec: torch.Tensor):
        model_params = {
            k: p.detach() if detach else p for k, p in model.named_parameters() if p.requires_grad
        }
        return flatten_dimensions(
            hvp(
                lambda p: batch_loss_function(model, loss)(p, x, y),
                model_params,
                align_structure(model_params, vec),
                reverse_only=reverse_only,
            ).values()
        )

    return b_hvp


def empirical_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    r"""
    Creates a function to compute the empirical loss of a given model on a given dataset.
    If we denote the model parameters with \( \theta \), the resulting function approximates:

    \[f(\theta) \coloneqq \frac{1}{N}\sum_{i=1}^N \operatorname{loss}(y_i, \operatorname{model}(\theta, x_i))\]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$,
    where $N$ is the number of all elements provided by the data_loader.

    Args:
      model: The model for which the loss should be computed.
      loss: The loss function to be used.
      data_loader: The data loader for iterating over the dataset.

    Returns:
        A function that computes the empirical loss of the model on the dataset for given model parameters.

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


def batch_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""
    Creates a function to compute the loss of a given model on a given batch of data, i.e. the function

    \[f(\theta, x, y) \coloneqq \frac{1}{N} \sum_{i=1}^N\operatorname{loss}(\operatorname{model}(\theta, x_i), y_i)\]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$,
    where $N$ is the number of elements in the batch.
    Args:
        model: The model for which the loss should be computed.
        loss: The loss function to be used, which should be able to handle a batch dimension

    Returns:
        A function that computes the loss of the model on a batch for given model parameters. The model
        parameter input to the function must take the form of a dict conform to model.named_parameters(), i.e. the keys
        must be a subset of the parameters and the corresponding tensor shapes must align. For the data input, the first
        dimension has to be the batch dimension.
    """

    def batch_loss(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        outputs = functional_call(model, params, (to_model_device(x, model),))
        return loss(outputs, y)

    return batch_loss


def get_hvp_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    use_hessian_avg: bool = True,
    reverse_only: bool = True,
    track_gradients: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that calculates the approximate Hessian-vector product for a given vector. If you want to
    compute the exact hessian, i.e., pulling all data into memory and compute a full gradient computation, use
    the function `hvp`.

    Args:
        model: A PyTorch module representing the model whose loss function's Hessian is to be computed.
        loss: A callable that takes the model's output and target as input and returns the scalar loss.
        data_loader: A DataLoader instance that provides batches of data for calculating the Hessian-vector product.
            Each batch from the DataLoader is assumed to return a tuple where the first element
            is the model's input and the second element is the target output.
        use_hessian_avg: If True, the returned function uses batch-wise Hessian computation via
            [batch_loss_function][pydvl.influence.torch.functional.batch_loss_function] and averages the results.
            If False, the function uses backpropagation on the full
            [empirical_loss_function][pydvl.influence.torch.functional.empirical_loss_function],
            which is more accurate than averaging the batch hessians, but probably has a way higher memory usage.
        reverse_only: Whether to use only reverse-mode autodiff (True, default) or
            both forward- and reverse-mode autodiff (False).
        track_gradients: Whether to track gradients for the resulting tensor of the hessian vector
            products (False, default).

    Returns:
        A function that takes a single argument, a vector, and returns the product of the Hessian of the `loss`
            function with respect to the `model`'s parameters and the input vector.
    """

    def hvp_function(vec: torch.Tensor) -> torch.Tensor:
        params = {
            k: p if track_gradients else p.detach()
            for k, p in model.named_parameters()
            if p.requires_grad
        }
        v = align_structure(params, vec)
        empirical_loss = empirical_loss_function(model, loss, data_loader)
        return flatten_dimensions(
            hvp(empirical_loss, params, v, reverse_only=reverse_only).values()
        )

    def avg_hvp_function(vec: torch.Tensor) -> torch.Tensor:
        num_batches = len(data_loader)
        avg_hessian = to_model_device(torch.zeros_like(vec), model)
        b_hvp = get_batch_hvp(model, loss, reverse_only)
        for x, y in iter(data_loader):
            avg_hessian += b_hvp(x, y, vec)

        return avg_hessian / float(num_batches)

    return avg_hvp_function if use_hessian_avg else hvp_function


def get_hessian(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    use_hessian_avg: bool = True,
    track_gradients: bool = False,
) -> torch.Tensor:
    """
    Computes the Hessian matrix for a given model and loss function.

    Args:
        model: The PyTorch model for which the Hessian is computed.
        loss: A callable that computes the loss.
        data_loader: DataLoader providing batches of input data and corresponding ground truths.
        use_hessian_avg: Flag to indicate whether the average Hessian across mini-batches should be computed.
                         If False, the empirical loss across the entire dataset is used.
        track_gradients: Whether to track gradients for the resulting tensor of the hessian vector
            products (False, default).

    Returns:
        A tensor representing the Hessian matrix. The shape of the tensor will be
        [num_parameters, num_parameters], where num_parameters is the number of trainable
        parameters in the model.
    """

    params = {
        k: p if track_gradients else p.detach()
        for k, p in model.named_parameters()
        if p.requires_grad
    }
    num_parameters = sum([p.numel() for p in params.values()])
    model_dtype = next((p.dtype for p in params.values()))
    flat_params = flatten_dimensions(params.values())

    if use_hessian_avg:
        hessian = torch.zeros((num_parameters, num_parameters), dtype=model_dtype)
        blf = batch_loss_function(model, loss)

        def flat_input_batch_loss_function(
            p: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor
        ):
            return blf(align_with_model(p, model), t_x, t_y)

        for x, y in iter(data_loader):

            hessian += torch.func.hessian(flat_input_batch_loss_function)(
                flat_params, x, y
            )

        hessian /= len(data_loader)
    else:

        def flat_input_empirical_loss(p: torch.Tensor):
            return empirical_loss_function(model, loss, data_loader)(
                align_with_model(p, model)
            )

        hessian = torch.func.jacrev(torch.func.jacrev(flat_input_empirical_loss))(
            flat_params
        )

    return hessian


def per_sample_loss(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Generates a function to compute per-sample losses using PyTorch's vmap, i.e. the vector-valued function

    \[f(\theta, x, y)  \coloneqq (\operatorname{loss}(\operatorname{model}(\theta, x_1), y_1), \dots,
    \operatorname{loss}(\operatorname{model}(\theta, x_N), y_N))\],

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$,
    where $N$ is the number of elements in the batch.

    Args:
        model: The PyTorch model for which per-sample losses will be computed.
        loss: A callable that computes the loss.

    Returns:
        A callable that computes the loss for each sample in the batch, given a dictionary of model
        inputs, the model's predictions, and the true values. The callable will return a tensor where
        each entry corresponds to the loss of the corresponding sample.
    """

    def compute_loss(
        params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        outputs = functional_call(
            model, params, (to_model_device(x.unsqueeze(0), model),)
        )
        return loss(outputs, y.unsqueeze(0))

    return torch.vmap(compute_loss, in_dims=(None, 0, 0))


def per_sample_gradient(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
]:
    """
    Generates a function to computes the per-sample gradient of the loss with respect to the model's parameters, i.e.
    the tensor-valued function

    \[
    f(\theta, x, y) /coloneqq (\nabla_{\theta}\operatorname{loss}(\operatorname{model}(\theta, x_1), y_1), \dots,
    \nabla_{\theta}\operatorname{loss}(\operatorname{model}(\theta, x_N), y_N)
    \]

    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$,
    where $N$ is the number of elements in the batch.

    Args:
        model: The PyTorch model for which per-sample gradients will be computed.
        loss: A callable that computes the loss.

    Returns:
        A callable that takes a dictionary of model parameters, the model's input, and the labels.
        It returns a dictionary with the same keys as the model's named parameters. Each entry in the
        returned dictionary corresponds to the gradient of the corresponding model parameter for each sample
        in the batch.

    """

    return torch.func.jacrev(per_sample_loss(model, loss))


def matrix_jacobian_product(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: torch.Tensor,
) -> Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Generates a function to computes the matrix-Jacobian product (MJP) of the per-sample loss with respect
    to the model's parameters, i.e. the function
    \[
    f(\theta, x, y) \coloneqg g @ (\nabla_{\theta}\operatorname{loss}(\operatorname{model}(\theta, x_i), y_i))_i^T
    \]
    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$.

    Args:
        model: The PyTorch model for which the MJP will be computed.
        loss: A callable that computes the loss.
        g: Matrix for which the product with the Jacobian will be computed. The shape of this matrix
           should be consistent with the shape of the jacobian.

    Returns:
        A callable that takes a dictionary of model inputs, the model's input, and the labels.
        The callable returns the matrix-Jacobian product of the per-sample loss with respect to the model's
        parameters for the given matrix `g`.

    """

    def single_jvp(
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        _g: torch.Tensor,
    ):
        return torch.func.jvp(
            lambda p: per_sample_loss(model, loss)(p, x, y),
            (params,),
            (align_with_model(_g, model),),
        )[1]

    def full_jvp(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        return torch.func.vmap(single_jvp, in_dims=(None, None, None, 0))(
            params, x, y, g
        )

    return full_jvp


def per_sample_mixed_derivative(
    model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
]:
    """
    Generates a function to computes the mixed derivatives, of the per-sample loss with respect
    to the model parameters and the input, i.e. the function
    \[
    f(\theta, x, y) \coloneqg \nabla_{\theta}\nabla_{x}$\operatorname{loss}$(\operatorname{model}(\theta, x), y)
    \]
    for a loss function $\operatorname{loss}$ and a model $\operatorname{model}$ with model parameters $\theta$.

    Args:
        model: The PyTorch model for which the mixed derivatives are computed.
        loss: A callable that computes the loss.
        g: Matrix for which the product with the Jacobian will be computed. The shape of this matrix
           should be consistent with the shape of the mixed jacobian.

    Returns:
        A callable that takes a dictionary of model inputs, the model's input, and the labels.
        The callable returns the mixed derivatives of the per-sample loss with respect to the model's
        parameters and input.

    """

    def compute_loss(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        outputs = functional_call(
            model, params, (to_model_device(x.unsqueeze(0), model),)
        )
        return loss(outputs, y.unsqueeze(0))

    return torch.vmap(
        torch.func.jacrev(torch.func.grad(compute_loss, argnums=1)),
        in_dims=(None, 0, 0),
    )
