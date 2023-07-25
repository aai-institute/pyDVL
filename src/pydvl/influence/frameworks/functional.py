from functools import partial
from typing import Callable, Dict, Generator, Iterable

import torch
from torch.func import functional_call, grad, jvp, vjp
from torch.utils.data import DataLoader

from .util import TensorContainerType, align_structure, to_model_device

__all__ = [
    "get_hvp_function",
]


def hvp(
    func: Callable[[TensorContainerType], TensorContainerType],
    params: TensorContainerType,
    vec: TensorContainerType,
    reverse_only: bool = True,
) -> TensorContainerType:
    """
     Computes the Hessian-vector product (HVP) for a given function at given parameters.
     This function can operate in two modes, either reverse-mode autodiff only or both
     forward- and reverse-mode autodiff.


    :param func: The function for which the HVP is computed.
    :param params: The parameters at which the HVP is computed.
    :param vec: The vector with which the Hessian is multiplied.
    :param reverse_only: Whether to use only reverse-mode autodiff
             (True, default) or both forward- and reverse-mode autodiff (False).

     :return: Input_type: The HVP of the function at the given parameters with the given vector.

     :Example:

      >>> def f(z): return torch.sum(z**2)
      >>> u = torch.ones(10, requires_grad=True)
      >>> v = torch.ones(10)
      >>> hvp_vec = hvp(f, u, v)
      >>> assert torch.allclose(hvp_vec, torch.full((10, ), 2.0))
    """
    output: TensorContainerType

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
) -> Generator[
    Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], None, None
]:
    """
    Generates a sequence of batch Hessian-vector product (HVP) computations for the provided model, loss function,
    and data loader.

    The generator iterates over the data_loader, creating partial function calls for calculating HVPs.

    :param model: The PyTorch model for which the HVP is calculated.
    :param loss: The loss function used to calculate the gradient and HVP.
    :param data_loader: PyTorch DataLoader object containing the dataset for which the HVP is calculated.
    :param reverse_only:
    :yield: A partial function H(vec)=hvp(model, loss, inputs, targets, vec) that when called,
            will compute the Hessian-vector product H(vec) for the given model, loss, inputs and targets.
    """

    for inputs, targets in iter(data_loader):
        batch_loss = batch_loss_function(model, loss, inputs, targets)
        yield partial(hvp, batch_loss, dict(model.named_parameters()), reverse_only=reverse_only)  # type: ignore


def empirical_loss_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Creates a function to compute the empirical loss of a given model on a given dataset.
    If we denote the model parameters with $\theta$, the resulting function approximates

    .. math::

        f(\theta) = \frac{1}{N}\sum_{i=1}^N \operatorname{loss}(y_i, \operatorname{model}(\theta, x_i)))

    :param model: The model for which the loss should be computed.
    :param loss: The loss function to be used.
    :param data_loader: The data loader for iterating over the dataset.

    :return: A function that computes the empirical loss
                  of the model on the dataset for given model parameters.

    """

    def empirical_loss(params: Dict[str, torch.Tensor]):
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
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
     Creates a function to compute the loss of a given model on a given batch of data.

    :param model: The model for which the loss should be computed.
    :param loss: The loss function to be used.
    :param x: The input data for the batch.
    :param y: The true labels for the batch.

    :return: A function that computes the loss
                   of the model on the batch for given model parameters.
    """

    def batch_loss(params: Dict[str, torch.Tensor]):
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
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Returns a function that calculates the approximate Hessian-vector product for a given vector. If you want to
    compute the exact hessian, i.e. pulling all data into memory and compute a full gradient computation, use
    the function :func:`hvp`.

    :param model: A PyTorch module representing the model whose loss function's Hessian is to be computed.
    :param loss: A callable that takes the model's output and target as input and returns the scalar loss.
    :param data_loader: A DataLoader instance that provides batches of data for calculating the Hessian-vector product.
                        Each batch from the DataLoader is assumed to return a tuple where the first element
                        is the model's input and the second element is the target output.
    :param use_hessian_avg: If True, it will use batch-wise Hessian computation. If False, the function averages
                            the batch gradients and perform backpropagation on the full (averaged) gradient,
                            which is more accurate than averaging the batch hessians,
                            but probably has a way higher memory usage.
    :param reverse_only: Whether to use only reverse-mode autodiff
            (True, default) or both forward- and reverse-mode autodiff (False)
    :return: A function that takes a single argument, a vector, and returns the product of the Hessian of the
             `loss` function with respect to the `model`'s parameters and the input vector.

    """
    params = dict(model.named_parameters())

    def hvp_function(vec: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        v = align_structure(params, vec)
        empirical_loss = empirical_loss_function(model, loss, data_loader)
        return hvp(empirical_loss, params, v, reverse_only=reverse_only)

    def avg_hvp_function(vec: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        v = align_structure(params, vec)
        batch_hessians: Iterable[Dict[str, torch.Tensor]] = map(
            lambda x: x(v), batch_hvp_gen(model, loss, data_loader, reverse_only)
        )

        result_dict = {
            key: to_model_device(torch.zeros_like(p), model)
            for key, p in params.items()
        }
        num_batches = len(data_loader)

        for batch_dict in batch_hessians:
            for key, value in batch_dict.items():
                result_dict[key] += value

        return {key: value / num_batches for key, value in result_dict.items()}

    return avg_hvp_function if use_hessian_avg else hvp_function
