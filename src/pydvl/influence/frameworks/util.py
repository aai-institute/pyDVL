import logging
import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generator, Iterable, Optional, Tuple, Union, Dict
import math

import torch
from torch.func import vjp, grad, jvp, functional_call
from numpy.typing import NDArray
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

Input_type = Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]


def to_model_device(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Returns the tensor `x` moved to the device of the `model`, if device of model is set
    :param x:
    :param model:
    :return:
    """
    if hasattr(model, "device"):
        return x.to(model.device)
    return x


def flatten_tensors_to_vector(tensors: Iterable[torch.Tensor]):
    """
    Flatten multiple tensors into a single 1D tensor (vector).

    The function takes an iterable of tensors and reshapes each of them into a 1D tensor.
    These reshaped tensors are then concatenated together into a single 1D tensor in the order they were given.

    Parameters:
    tensors (Iterable[torch.Tensor]): An iterable of tensors to be reshaped and concatenated.

    Returns:
    torch.Tensor: A 1D tensor that is the concatenation of all the reshaped input tensors.
    """

    return torch.cat([t.contiguous().view(-1) for t in tensors])


def reshape_vector_to_tensors(input_vector: torch.Tensor, target_shapes: Iterable[Tuple[int, ...]]):
    """
    Reshape a 1D tensor into multiple tensors with specified shapes.

    The function takes a 1D tensor (input_vector) and reshapes it into a series of tensors with shapes given by 'target_shapes'.
    The reshaped tensors are returned as a tuple in the same order as their corresponding shapes.

    Note: The total number of elements in 'input_vector' must be equal to the sum of the products of the shapes in 'target_shapes'.

    Parameters:
    input_vector (Tensor): The 1D tensor to be reshaped. Must be 1D.
    target_shapes (Iterable[Tuple[int, ...]]): An iterable of tuples. Each tuple defines the shape of a tensor to be
                                                  reshaped from the 'input_vector'.

    Returns:
    tuple[torch.Tensor]: A tuple of reshaped tensors.

    Raises:
    ValueError: If 'input_vector' is not a 1D tensor or if the total number of elements in 'input_vector' does not match
                the sum of the products of the shapes in 'target_shapes'.
    """

    if input_vector.dim() != 1:
        raise ValueError("Input vector must be a 1D tensor")

    total_elements = sum(math.prod(shape) for shape in target_shapes)

    if total_elements != input_vector.shape[0]:
        raise ValueError(f"The total elements in shapes {total_elements} does not match the vector length {input_vector.shape[0]}")

    tensors = []
    start = 0
    for shape in target_shapes:
        size = math.prod(shape)  # compute the total size of the tensor with this shape
        tensors.append(input_vector[start:start+size].view(shape))  # slice the vector and reshape it
        start += size
    return tuple(tensors)


def hvp(
        func: Callable[[Input_type], torch.Tensor],
        params: Input_type,
        vec: Input_type,
        reverse_only: bool = True
) -> Input_type:
    """
    Computes the Hessian-vector product (HVP) for a given function at given parameters.
    This function can operate in two modes, either reverse-mode autodiff only or both
    forward- and reverse-mode autodiff.


    Args:
        func (Callable[[Input_type], torch.Tensor]): The function for which the HVP is computed.
        params (Input_type): The parameters at which the HVP is computed.
        vec (Input_type): The vector with which the Hessian is multiplied.
        reverse_only (bool, optional): Whether to use only reverse-mode autodiff
            (True, default) or both forward- and reverse-mode autodiff (False).

    Returns:
        Input_type: The HVP of the function at the given parameters with the given vector.

    Examples:
        >>> def f(z): return torch.sum(z**2)
        >>> u = torch.ones(10, requires_grad=True)
        >>> v = torch.ones(10)
        >>> hvp_vec = hvp(f, u, v)
        >>> assert torch.allclose(hvp_vec, torch.full((10, ), 2.0))
    """

    if reverse_only:
        _, vjp_fn = vjp(grad(func), params)
        output = vjp_fn(vec)[0]
    else:
        output = jvp(grad(func), (params, ), (vec,))[1]

    return output



def batch_hvp_gen(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    reverse_only: bool = True
) -> Generator[Callable[[torch.Tensor], torch.Tensor], None, None]:
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
        yield partial(hvp, batch_loss, dict(model.named_parameters()), reverse_only=reverse_only)


def get_hvp_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    use_hessian_avg: bool = True,
    reverse_only: bool = True
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

        result_dict = {key: to_model_device(torch.zeros_like(p), model) for key, p in params.items()}
        num_batches = len(data_loader)

        for batch_dict in batch_hessians:
            for key, value in batch_dict.items():
                result_dict[key] += value

        return {key: value/num_batches for key, value in result_dict.items()}

    return avg_hvp_function if use_hessian_avg else hvp_function


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


def lanzcos_low_rank_hessian_approx(
    hessian_vp: Callable[[torch.Tensor], torch.Tensor],
    matrix_shape: Tuple[int, int],
    hessian_perturbation: float = 0.0,
    rank_estimate: int = 10,
    krylov_dimension: Optional[int] = None,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    device: Optional[torch.device] = None,
    eigen_computation_on_gpu: bool = False,
) -> LowRankProductRepresentation:
    """
    Calculates a low-rank approximation of the Hessian matrix of the model's loss function using the implicitly
    restarted Lanczos algorithm.


    :param hessian_vp: A function that takes a vector and returns the product of the Hessian of the loss function
    :param matrix_shape: The shape of the matrix, represented by hessian vector product.
    :param hessian_perturbation: Optional regularization parameter added to the Hessian-vector product
                                 for numerical stability.
    :param rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
                          Represents the desired rank of the Hessian approximation.
    :param krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
                             If not provided, it defaults to $min(model.num_parameters, max(2*rank_estimate + 1, 20))$.
    :param x0: An optional initial vector to use in the Lanczos algorithm.
               If not provided, a random initial vector is used.
    :param tol: The stopping criteria for the Lanczos algorithm, which stops when the difference
                in the approximated eigenvalue is less than `tol`. Defaults to 1e-6.
    :param max_iter: The maximum number of iterations for the Lanczos method. If not provided, it defaults to
                     $10*model.num_parameters$
    :param device: The device to use for executing the hessian vector product.
    :param eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the provided
                                     device via cupy implementation.
                                     Make sure, that either your model is small enough or you use a
                                     small rank_estimate to fit your device's memory.
                                     If False, the eigen pair approximation is executed on the CPU by scipy wrapper to
                                     ARPACK.
    :return: A `LowRankProductRepresentation` instance that contains the top (up until rank_estimate) eigenvalues
             and corresponding eigenvectors of the Hessian.
    """

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

        def mv_cupy(x):
            x = from_dlpack(x.toDlpack())
            y = hessian_vp(x) + hessian_perturbation * x
            return cp.from_dlpack(to_dlpack(y))

        eigen_vals, eigen_vecs = eigsh(
            LinearOperator(matrix_shape, matvec=mv_cupy),
            k=rank_estimate,
            maxiter=max_iter,
            tol=tol,
            ncv=krylov_dimension,
            return_eigenvectors=True,
        )
        return LowRankProductRepresentation(
            from_dlpack(eigen_vals.toDlpack()), from_dlpack(eigen_vecs.toDlpack())
        )

    else:
        from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
        torch_default_dtype = torch.get_default_dtype()

        def mv_scipy(x: NDArray) -> NDArray:
            x_torch = torch.as_tensor(x, device=device, dtype=torch_default_dtype)
            y: NDArray = (
                (hessian_vp(x_torch) + hessian_perturbation * x_torch).detach().cpu().numpy()
            )
            return y

        try:
            eigen_vals, eigen_vecs = eigsh(
                A=LinearOperator(matrix_shape, matvec=mv_scipy),
                k=rank_estimate,
                maxiter=max_iter,
                tol=tol,
                ncv=krylov_dimension,
                return_eigenvectors=True,
                v0=x0.cpu().numpy() if x0 is not None else None,
            )
        except ArpackNoConvergence as e:
            logger.warning(
                f"ARPACK did not converge for parameters {max_iter=}, {tol=}, {krylov_dimension=}, "
                f"{rank_estimate=}. \n Returning the best approximation found so far. Use those with care or "
                f"modify parameters.\n Original error: {e}"
            )
            return LowRankProductRepresentation(
                torch.as_tensor(e.eigenvalues, dtype=torch_default_dtype),
                torch.as_tensor(e.eigenvectors, dtype=torch_default_dtype)
            )

        return LowRankProductRepresentation(
            torch.as_tensor(eigen_vals, dtype=torch_default_dtype),
            torch.as_tensor(eigen_vecs, dtype=torch_default_dtype)
        )


def empirical_loss_function(model: torch.nn.Module,
                            loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            data_loader: DataLoader) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Creates a function to compute the empirical loss of a given model on a given dataset.
    If we denote the model parameters with $\theta$, the resulting function approximates

    .. math::

        f(\theta) = \frac{1}{N}\sum_{i=1}^N \operatorname{loss}(y_i, \operatorname{model}(\theta, x_i)))

    Args:
        model (torch.nn.Module): The model for which the loss should be computed.
        loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to be used.
        data_loader (torch.utils.data.DataLoader): The data loader for iterating over the dataset.

    Returns:
        Callable[[torch.nn.parameter.Parameter], torch.Tensor]: A function that computes the empirical loss
                                                               of the model on the dataset for given model parameters.

    """
    def empirical_loss(params: Dict[str, torch.Tensor]):
        total_loss = to_model_device(torch.zeros((), requires_grad=True), model)
        total_samples = to_model_device(torch.zeros(()), model)

        for x, y in iter(data_loader):
            output = functional_call(model, params, (to_model_device(x, model),), strict=True)
            loss_value = loss(output, to_model_device(y, model))
            total_loss = total_loss + loss_value * x.size(0)
            total_samples += x.size(0)

        return total_loss / total_samples

    return empirical_loss


def batch_loss_function(model: torch.nn.Module,
                        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        x: torch.Tensor,
                        y: torch.Tensor) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Creates a function to compute the loss of a given model on a given batch of data.

    Args:
        model (torch.nn.Module): The model for which the loss should be computed.
        loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to be used.
        x (torch.Tensor): The input data for the batch.
        y (torch.Tensor): The true labels for the batch.

    Returns:
        Callable[[Dict[str, torch.Tensor]], torch.Tensor]: A function that computes the loss
                                                           of the model on the batch for given model parameters.
    """
    def batch_loss(params: Dict[str, torch.Tensor]):
        outputs = functional_call(model, params, (to_model_device(x, model),), strict=True)
        return loss(outputs, y)

    return batch_loss


Input_type = Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]


def align_structure(source: Dict[str, torch.Tensor], target: Input_type):
    """
    This function transforms `target` to have the same structure as `source`, i.e.,
    it should be a dictionary with the same keys as `source` and each corresponding
    value in `target` should have the same shape as the value in `source`.

    Args:
        source (dict): The reference dictionary containing PyTorch tensors.
        target (Input_type): The input to be harmonized. It can be a dictionary, tuple, or tensor.

    Returns:
        dict: The harmonized version of `target`.

    Raises:
        ValueError: If `target` cannot be harmonized to match `source`.
    """

    tangent = copy.copy(target)

    if isinstance(tangent, dict):
        if list(tangent.keys()) != list(source.keys()):
            raise ValueError("The keys in 'target' do not match the keys in 'source'.")
        if list(map(lambda v: v.shape, tangent.values())) != list(map(lambda v: v.shape, source.values())):
            raise ValueError("The shapes of the values in 'target' do not match the shapes of the values in 'source'.")
    elif isinstance(tangent, tuple) or isinstance(tangent, list):
        if list(map(lambda v: v.shape, tangent)) != list(map(lambda v: v.shape, source.values())):
            raise ValueError("'target' is a tuple/list but its elements' shapes do not match the shapes "
                             "of the values in 'source'.")
        tangent = dict(zip(source.keys(), tangent))
    elif isinstance(tangent, torch.Tensor):
        try:
            tangent = reshape_vector_to_tensors(tangent, list(map(lambda p: p.shape, source.values())))
            tangent = dict(zip(source.keys(), tangent))
        except Exception as e:
            raise ValueError(f"'target' is a tensor but cannot be reshaped to match 'source'. Original error: {e}")
    else:
        raise ValueError(f"'target' is of type {type(tangent)} which is not supported.")

    return tangent
