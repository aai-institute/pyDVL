import copy
import logging
import math
from typing import Dict, Iterable, Tuple, Union

import torch

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


def reshape_vector_to_tensors(
    input_vector: torch.Tensor, target_shapes: Iterable[Tuple[int, ...]]
):
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
        raise ValueError(
            f"The total elements in shapes {total_elements} does not match the vector length {input_vector.shape[0]}"
        )

    tensors = []
    start = 0
    for shape in target_shapes:
        size = math.prod(shape)  # compute the total size of the tensor with this shape
        tensors.append(
            input_vector[start : start + size].view(shape)
        )  # slice the vector and reshape it
        start += size
    return tuple(tensors)


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
        if list(map(lambda v: v.shape, tangent.values())) != list(
            map(lambda v: v.shape, source.values())
        ):
            raise ValueError(
                "The shapes of the values in 'target' do not match the shapes of the values in 'source'."
            )
    elif isinstance(tangent, tuple) or isinstance(tangent, list):
        if list(map(lambda v: v.shape, tangent)) != list(
            map(lambda v: v.shape, source.values())
        ):
            raise ValueError(
                "'target' is a tuple/list but its elements' shapes do not match the shapes "
                "of the values in 'source'."
            )
        tangent = dict(zip(source.keys(), tangent))
    elif isinstance(tangent, torch.Tensor):
        try:
            tangent = reshape_vector_to_tensors(
                tangent, list(map(lambda p: p.shape, source.values()))
            )
            tangent = dict(zip(source.keys(), tangent))
        except Exception as e:
            raise ValueError(
                f"'target' is a tensor but cannot be reshaped to match 'source'. Original error: {e}"
            )
    else:
        raise ValueError(f"'target' is of type {type(tangent)} which is not supported.")

    return tangent
