import logging
import math
from typing import Any, Collection, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "to_model_device",
    "reshape_vector_to_tensors",
    "TorchTensorContainerType",
    "align_structure",
    "as_tensor",
    "align_with_model",
    "flatten_dimensions",
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


def align_with_model(x: TorchTensorContainerType, model: torch.nn.Module):
    """
    Aligns an input to the model's parameter structure, i.e. transforms it into a dict with the same keys as
    model.named_parameters() and matching tensor shapes

    Args:
        x: The input to be aligned. It can be a dictionary, tuple, or tensor.
        model: model to use for alignment

    Returns:
        The aligned version of `x`.

    Raises:
        ValueError: If `x` cannot be aligned to match the model's parameters .

    """
    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    return align_structure(model_params, x)


def flatten_dimensions(
    tensors: Iterable[torch.Tensor],
    shape: Optional[Tuple[int, ...]] = None,
    concat_at: int = -1,
) -> torch.Tensor:
    """
    Flattens the dimensions of each tensor in the given iterable and concatenates them along a specified dimension.

    This function takes an iterable of PyTorch tensors and flattens each tensor.
    Optionally, each tensor can be reshaped to a specified shape before concatenation.
    The concatenation is performed along the dimension specified by `concat_at`.

    Args:
        tensors: An iterable containing PyTorch tensors to be flattened and concatenated.
        shape: A tuple representing the desired shape to which each tensor is reshaped before concatenation.
            If None, tensors are flattened to 1D. Defaults to None.
        concat_at: The dimension along which to concatenate the tensors. Defaults to -1.

    Returns:
        A single tensor resulting from the concatenation of the input tensors,
        each either flattened or reshaped as specified.

    Examples:
        >>> tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]
        >>> flatten_dimensions(tensors)
        tensor([1, 2, 3, 4, 5, 6, 7, 8])

        >>> flatten_dimensions(tensors, shape=(2, 2), concat_at=0)
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])
    """
    return torch.cat(
        [t.reshape(-1) if shape is None else t.reshape(*shape) for t in tensors],
        dim=concat_at,
    )
