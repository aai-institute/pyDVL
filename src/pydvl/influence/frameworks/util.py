import logging
from dataclasses import dataclass
from functools import partial, reduce
from typing import Callable, Generator, Iterable, Optional, Tuple

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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


def hvp(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """
    Returns H*vec where H is the Hessian of the loss with respect to
    the model parameters.

    :param model: A torch.nn.Module, whose parameters are used for backpropagation.
    :param loss: A loss function, which is a callable that takes the model's
                 output and target as input and returns a scalar loss.
    :param x: Input tensor to the model.
    :param y: Target output tensor.
    :param vec: The vector with which the Hessian of the loss function is to be multiplied.
    :return: A tensor of the same shape as vec, representing the product of the Hessian of the loss function
             with respect to the model parameters and the input vector.
    """
    outputs = model(to_model_device(x, model))
    loss_value = loss(outputs, to_model_device(y, model))
    params = [p for p in model.parameters() if p.requires_grad]

    grads = torch.autograd.grad(loss_value, params, create_graph=True)

    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    grad_vec_product = torch.dot(flat_grads, vec)
    hessian_vec_prod = torch.autograd.grad(grad_vec_product, params)
    hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod])
    return hessian_vec_prod


def batch_hvp_gen(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> Generator[Callable[[torch.Tensor], torch.Tensor], None, None]:
    """
    Generates a sequence of batch Hessian-vector product (HVP) computations for the provided model, loss function,
    and data loader.

    The generator iterates over the data_loader, creating partial function calls for calculating HVPs.

    :param model: The PyTorch model for which the HVP is calculated.
    :param loss: The loss function used to calculate the gradient and HVP.
    :param data_loader: PyTorch DataLoader object containing the dataset for which the HVP is calculated.
    :yield: A partial function H(vec)=hvp(model, loss, inputs, targets, vec) that when called,
            will compute the Hessian-vector product H(vec) for the given model, loss, inputs and targets.
    """

    for inputs, targets in iter(data_loader):
        yield partial(hvp, model, loss, inputs, targets)


def avg_gradient(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
) -> torch.Tensor:
    """
    Returns the average gradient of the loss function with respect to the model parameters
    :param model: A torch.nn.Module, whose parameters are used for backpropagation.
    :param loss: A loss function, which is a callable that takes the model's output and target as input
    and returns a scalar loss.
    :param data_loader: an instance of :class:`torch.utils.data.DataLoader`
    :return: average of batch gradients
    """
    params = [p for p in model.parameters() if p.requires_grad]
    total_grad_xy = None
    total_points = 0
    num_batches = len(data_loader)

    for k, (x, y) in enumerate(data_loader):
        logger.debug(f"Computing the gradient for batch {k+1}/{num_batches}")
        outputs = model(to_model_device(x, model))
        loss_value = loss(outputs, to_model_device(y, model))
        grads = torch.autograd.grad(loss_value, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

        num_points_in_batch = len(x)
        if total_grad_xy is None:
            total_grad_xy = flat_grads * num_points_in_batch
        else:
            total_grad_xy += flat_grads * num_points_in_batch
        total_points += num_points_in_batch

    if total_grad_xy is None:
        raise ValueError("DataLoader is empty")

    return total_grad_xy / total_points


def get_hvp_function(
    model: torch.nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    use_hessian_avg: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
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

    def hvp_function(vec: torch.Tensor) -> torch.Tensor:
        params = [p for p in model.parameters() if p.requires_grad]
        avg_grad = avg_gradient(model, loss, data_loader)

        grad_vec_product = torch.dot(avg_grad, vec)
        hessian_vec_prod = torch.autograd.grad(grad_vec_product, params)
        hessian_vec_prod = torch.cat(
            [g.contiguous().view(-1) for g in hessian_vec_prod]
        )
        return hessian_vec_prod

    def avg_hvp_function(vec: torch.Tensor) -> torch.Tensor:
        batch_hessians: Iterable[torch.Tensor] = map(
            lambda x: x(vec), batch_hvp_gen(model, loss, data_loader)
        )
        return reduce(lambda x, y: x + y, batch_hessians) / len(data_loader)

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

        def mv_scipy(x: NDArray) -> NDArray:
            x_torch = torch.from_numpy(x)
            if device is not None:
                x_torch = x_torch.to(device)
            y: NDArray = (
                (hessian_vp(x_torch) + hessian_perturbation * x_torch).cpu().numpy()
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
                torch.from_numpy(e.eigenvalues), torch.from_numpy(e.eigenvectors)
            )

        return LowRankProductRepresentation(
            torch.from_numpy(eigen_vals), torch.from_numpy(eigen_vecs)
        )
