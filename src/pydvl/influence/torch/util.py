import logging
import math
from typing import Any, Collection, Dict, Iterable, Mapping, Tuple, Union

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "to_model_device",
    "flatten_tensors_to_vector",
    "reshape_vector_to_tensors",
    "TorchTensorContainerType",
    "align_structure",
    "as_tensor",
]


def to_model_device(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Returns the tensor `x` moved to the device of the `model`, if device of model is set

    Args:
        x: The tensor to be moved to the device of the model.
        model: The model whose device will be used to move the tensor.

    Returns:
        The tensor `x` moved to the device of the `model`, if device of model is set.
    """
    device = next(model.parameters()).device
    return x.to(device)


def flatten_tensors_to_vector(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Flatten multiple tensors into a single 1D tensor (vector).

    This function takes an iterable of tensors and reshapes each of them into a 1D tensor.
    These reshaped tensors are then concatenated together into a single 1D tensor in the order they were given.

    Args:
        tensors: An iterable of tensors to be reshaped and concatenated.

    Returns:
        A 1D tensor that is the concatenation of all the reshaped input tensors.
    """
    return torch.cat([t.contiguous().view(-1) for t in tensors])


def reshape_vector_to_tensors(
    input_vector: torch.Tensor, target_shapes: Iterable[Tuple[int, ...]]
) -> Tuple[torch.Tensor, ...]:
    """
    Reshape a 1D tensor into multiple tensors with specified shapes.

    This function takes a 1D tensor (input_vector) and reshapes it into a series of tensors with shapes given by 'target_shapes'.
    The reshaped tensors are returned as a tuple in the same order as their corresponding shapes.

    Note: The total number of elements in 'input_vector' must be equal to the sum of the products of the shapes in 'target_shapes'.

    Args:
        input_vector: The 1D tensor to be reshaped. Must be 1D.
        target_shapes: An iterable of tuples. Each tuple defines the shape of a tensor to be reshaped from the 'input_vector'.

    Returns:
        A tuple of reshaped tensors.

    Raises:
        ValueError: If 'input_vector' is not a 1D tensor or if the total number of elements in 'input_vector' does not match the sum of the products of the shapes in 'target_shapes'.
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


TorchTensorContainerType = Union[
    torch.Tensor,
    Collection[torch.Tensor],
    Mapping[str, torch.Tensor],
]
"""Type for a PyTorch tensor or a container thereof."""


def align_structure(
    source: Mapping[str, torch.Tensor],
    target: TorchTensorContainerType,
) -> Dict[str, torch.Tensor]:
    """
    This function transforms `target` to have the same structure as `source`, i.e.,
    it should be a dictionary with the same keys as `source` and each corresponding
    value in `target` should have the same shape as the value in `source`.

    Args:
        source: The reference dictionary containing PyTorch tensors.
        target: The input to be harmonized. It can be a dictionary, tuple, or tensor.

    Returns:
        The harmonized version of `target`.

    Raises:
        ValueError: If `target` cannot be harmonized to match `source`.
    """

    tangent_dict: Dict[str, torch.Tensor]

    if isinstance(target, dict):

        if list(target.keys()) != list(source.keys()):
            raise ValueError("The keys in 'target' do not match the keys in 'source'.")

        if [v.shape for v in target.values()] != [v.shape for v in source.values()]:

            raise ValueError(
                "The shapes of the values in 'target' do not match the shapes of the values in 'source'."
            )

        tangent_dict = target

    elif isinstance(target, tuple) or isinstance(target, list):

        if [v.shape for v in target] != [v.shape for v in source.values()]:

            raise ValueError(
                "'target' is a tuple/list but its elements' shapes do not match the shapes "
                "of the values in 'source'."
            )

        tangent_dict = dict(zip(source.keys(), target))

    elif isinstance(target, torch.Tensor):

        try:
            tangent_dict = dict(
                zip(
                    source.keys(),
                    reshape_vector_to_tensors(
                        target, [p.shape for p in source.values()]
                    ),
                )
            )
        except Exception as e:
            raise ValueError(
                f"'target' is a tensor but cannot be reshaped to match 'source'. Original error: {e}"
            )

    else:
        raise ValueError(f"'target' is of type {type(target)} which is not supported.")

    return tangent_dict


def as_tensor(a: Any, warn=True, **kwargs) -> torch.Tensor:
    """
    Converts an array into a torch tensor.

    Args:
        a: Array to convert to tensor.
        warn: If True, warns that `a` will be converted.

    Returns:
        A torch tensor converted from the input array.
    """

    if warn and not isinstance(a, torch.Tensor):
        logger.warning("Converting tensor to type torch.Tensor.")
    return torch.as_tensor(a, **kwargs)
