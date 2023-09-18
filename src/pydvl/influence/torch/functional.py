from typing import Callable, Collection, Dict, Generator, Iterable, Mapping, Tuple, cast

import torch
from torch.func import functional_call, grad, jvp, vjp
from torch.utils.data import DataLoader

from .util import (
    TorchTensorContainerType,
    align_structure,
    flatten_tensors_to_vector,
    to_model_device,
)

__all__ = [
    "get_hvp_function",
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

    ??? Example

        ```pycon
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


def batch_hvp_gen(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    reverse_only: bool = True,
) -> Generator[Callable[[TorchTensorContainerType], torch.Tensor], None, None]:
    r"""
    Generates a sequence of batch Hessian-vector product (HVP) computations for the provided model, loss function,
    and data loader. If \(f_i\) is the model's loss on the \(i\)-th batch and \(\theta\) the model parameters,
    this is the sequence of the callable matrix vector products for the matrices

    \[\nabla_{\theta}\nabla_{\theta}f_i(\theta), \quad i=1,\dots, \text{num_batches} \]

    i.e. iterating over the data_loader, yielding partial function calls for calculating HVPs.

    Args:
        model: The PyTorch model for which the HVP is calculated.
        loss: The loss function used to calculate the gradient and HVP.
        data_loader: PyTorch DataLoader object containing the dataset for which the HVP is calculated.
        reverse_only: Whether to use only reverse-mode autodiff
            (True, default) or both forward- and reverse-mode autodiff (False).

    Yields:
        Partial functions `H_{batch}(vec)=hvp(model, loss, inputs, targets, vec)` that when called,
            will compute the Hessian-vector product H(vec) for the given model and loss in a batch-wise manner, where
            (inputs, targets) coming from one batch.
    """

    for inputs, targets in iter(data_loader):
        batch_loss = batch_loss_function(model, loss, inputs, targets)
        model_params = dict(model.named_parameters())

        def batch_hvp(vec: TorchTensorContainerType) -> torch.Tensor:
            aligned_params = align_structure(model_params, vec)
            hvp_result = hvp(
                batch_loss,
                model_params,
                aligned_params,
                reverse_only=reverse_only,
            )
            hvp_result_values: Collection[torch.Tensor]
            if isinstance(hvp_result, Mapping):
                hvp_result_values = cast(
                    Collection[torch.Tensor], tuple(hvp_result.values())
                )
            elif isinstance(hvp_result, Collection):
                hvp_result_values = cast(Collection[torch.Tensor], hvp_result)
            else:
                hvp_result_values = [hvp_result]
            return flatten_tensors_to_vector(hvp_result_values)

        yield batch_hvp


def empirical_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> Callable[[TorchTensorContainerType], torch.Tensor]:
    r"""
    Creates a function to compute the empirical loss of a given model on a given dataset.
    If we denote the model parameters with \( \theta \), the resulting function approximates:

    \[f(\theta) = \frac{1}{N}\sum_{i=1}^N \operatorname{loss}(y_i, \operatorname{model}(\theta, x_i))\]

    Args:
        model: The model for which the loss should be computed.
        loss: The loss function to be used.
        data_loader: The data loader for iterating over the dataset.

    Returns:
        A function that computes the empirical loss of the model on the dataset for given model parameters.

    """

    def empirical_loss(params: TorchTensorContainerType) -> torch.Tensor:
        total_loss = to_model_device(torch.zeros((), requires_grad=True), model)
        total_samples = to_model_device(torch.zeros(()), model)

        for x, y in iter(data_loader):
            output = functional_call(
                model, params, (to_model_device(x, model),), strict=True
            )
            loss_value = loss(output, to_model_device(y, model))
            total_loss = total_loss + loss_value * x.size(0)
            total_samples += x.size(0)

        return total_loss / total_samples

    return empirical_loss


def batch_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> Callable[[TorchTensorContainerType], torch.Tensor]:
    r"""
    Creates a function to compute the loss of a given model on a given batch of data, i.e. for the $i$-th batch $B_i$

    \[\frac{1}{|B_i|}\sum_{x,y \in B_i} \operatorname{loss}(y, \operatorname{model}(\theta, x))\]

    Args:
        model: The model for which the loss should be computed.
        loss: The loss function to be used.
        x: The input data for the batch.
        y: The true labels for the batch.

    Returns:
        A function that computes the loss of the model on the batch for given model parameters.
    """

    def batch_loss(params: TorchTensorContainerType) -> torch.Tensor:
        outputs = functional_call(
            model, params, (to_model_device(x, model),), strict=True
        )
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
            products are (False, default).

    Returns:
        A function that takes a single argument, a vector, and returns the product of the Hessian of the `loss`
            function with respect to the `model`'s parameters and the input vector.
    """

    params = {
        k: p if track_gradients else p.detach() for k, p in model.named_parameters()
    }

    def hvp_function(vec: torch.Tensor) -> torch.Tensor:
        aligned_vec = align_structure(params, vec)
        empirical_loss = empirical_loss_function(model, loss, data_loader)
        hvp_result = hvp(empirical_loss, params, aligned_vec, reverse_only=reverse_only)
        hvp_result_values: Collection[torch.Tensor]
        if isinstance(hvp_result, Mapping):
            hvp_result_values = cast(
                Collection[torch.Tensor], tuple(hvp_result.values())
            )
        elif isinstance(hvp_result, Collection):
            hvp_result_values = cast(Collection[torch.Tensor], hvp_result)
        else:
            hvp_result_values = [hvp_result]
        return flatten_tensors_to_vector(hvp_result_values)

    def avg_hvp_function(vec: torch.Tensor) -> torch.Tensor:
        aligned_vec = align_structure(params, vec)
        batch_hessians_vector_products: Iterable[torch.Tensor] = map(
            lambda x: x(aligned_vec),
            batch_hvp_gen(model, loss, data_loader, reverse_only),
        )

        num_batches = len(data_loader)
        avg_hessian = to_model_device(torch.zeros_like(vec), model)

        for batch_hvp in batch_hessians_vector_products:
            avg_hessian += batch_hvp

        return avg_hessian / float(num_batches)

    return avg_hvp_function if use_hessian_avg else hvp_function
