import logging
import math
from functools import partial
from typing import (
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import dask
import numpy as np
import torch
from dask import array as da
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..array import NumpyConverter, TensorAggregator

logger = logging.getLogger(__name__)

__all__ = [
    "to_model_device",
    "reshape_vector_to_tensors",
    "TorchTensorContainerType",
    "align_structure",
    "align_with_model",
    "flatten_dimensions",
    "TorchCatAggregator",
    "TorchNumpyConverter",
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

    This function takes a 1D tensor (input_vector) and reshapes it into a series of tensors with shapes given by
    'target_shapes'. The reshaped tensors are returned as a tuple in the same order as their corresponding shapes.

    Note: The total number of elements in 'input_vector' must be equal to the sum of the products of the shapes
        in 'target_shapes'.

    Args:
        input_vector: The 1D tensor to be reshaped. Must be 1D.
        target_shapes: An iterable of tuples. Each tuple defines the shape of a tensor to be reshaped from the
            'input_vector'.

    Returns:
        A tuple of reshaped tensors.

    Raises:
        ValueError: If 'input_vector' is not a 1D tensor or if the total number of elements in 'input_vector' does not
            match the sum of the products of the shapes in 'target_shapes'.
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

    ??? Example
        ```pycon
        >>> tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]
        >>> flatten_dimensions(tensors)
        tensor([1, 2, 3, 4, 5, 6, 7, 8])

        >>> flatten_dimensions(tensors, shape=(2, 2), concat_at=0)
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])
        ```
    """
    return torch.cat(
        [t.reshape(-1) if shape is None else t.reshape(*shape) for t in tensors],
        dim=concat_at,
    )


def torch_dataset_to_dask_array(
    dataset: Dataset,
    chunk_size: int,
    total_size: Optional[int] = None,
    resulting_dtype=np.float32,
) -> Tuple[da.Array, ...]:
    """
    Construct tuple of dask arrays from a PyTorch dataset, using dask.delayed

    Args:
        dataset: A PyTorch [dataset][torch.utils.data.Dataset]
        chunk_size: The size of the chunks for the resulting Dask arrays.
        total_size: If the dataset does not implement len, provide the length via this parameter. If None
            the length of the dataset is inferred via accessing the dataset once.
        resulting_dtype: The dtype of the resulting [dask.array.Array][dask.array.Array]

    ??? Example
        ```python
        import torch
        from torch.utils.data import TensorDataset
        x = torch.rand((20, 3))
        y = torch.rand((20, 1))
        dataset = TensorDataset(x, y)
        da_x, da_y = torch_dataset_to_dask_array(dataset, 4)
        ```

    Returns:
        Tuple of Dask arrays corresponding to each tensor in the dataset.
    """

    def _infer_data_len(d_set: Dataset):
        try:
            n_data = len(d_set)
            if total_size is not None and n_data != total_size:
                raise ValueError(
                    f"The number of samples in the dataset ({n_data}), derived from calling ´len´, "
                    f"does not match the provided total number of samples ({total_size}). Call"
                    f"the function without total_size."
                )
            return n_data
        except TypeError as e:
            err_msg = f"Could not infer the number of samples in the dataset from calling ´len´. Original error: {e}."
            if total_size is not None:
                logger.warning(
                    err_msg
                    + f" Using the provided total number of samples {total_size}."
                )
                return total_size
            else:
                logger.warning(
                    err_msg
                    + f" Infer the number of samples from the dataset, via iterating the dataset once. "
                    f"This might induce severe overhead, so consider"
                    f"providing total_size, if you know the number of samples beforehand."
                )
                idx = 0
                while True:
                    try:
                        t = d_set[idx]
                        if all(_t.numel() == 0 for _t in t):
                            return idx
                        idx += 1

                    except IndexError:
                        return idx

    sample = dataset[0]
    if not isinstance(sample, tuple):
        sample = (sample,)

    def _get_chunk(
        start_idx: int, stop_idx: int, d_set: Dataset
    ) -> Tuple[torch.Tensor, ...]:
        try:
            t = d_set[start_idx:stop_idx]
            if not isinstance(t, tuple):
                t = (t,)
            return t  # type:ignore
        except Exception:
            nested_tensor_list = [
                [d_set[idx][k] for idx in range(start_idx, stop_idx)]
                for k in range(len(sample))
            ]
            return tuple(map(torch.stack, nested_tensor_list))

    n_samples = _infer_data_len(dataset)
    chunk_indices = [
        (i, min(i + chunk_size, n_samples)) for i in range(0, n_samples, chunk_size)
    ]
    delayed_dataset = dask.delayed(dataset)
    delayed_chunks = [
        dask.delayed(partial(_get_chunk, start, stop))(delayed_dataset)
        for (start, stop) in chunk_indices
    ]

    delayed_arrays_dict: Dict[int, List[da.Array]] = {k: [] for k in range(len(sample))}

    for chunk, (start, stop) in zip(delayed_chunks, chunk_indices):
        for tensor_idx, sample_tensor in enumerate(sample):

            delayed_tensor = da.from_delayed(
                dask.delayed(lambda t: t.cpu().numpy())(chunk[tensor_idx]),
                shape=(stop - start, *sample_tensor.shape),
                dtype=resulting_dtype,
            )

            delayed_arrays_dict[tensor_idx].append(delayed_tensor)

    return tuple(
        da.concatenate(array_list) for array_list in delayed_arrays_dict.values()
    )


class TorchNumpyConverter(NumpyConverter[torch.Tensor]):
    """
    Helper class for converting between [torch.Tensor][torch.Tensor] and [numpy.ndarray][numpy.ndarray]

    Args:
        device: Optional device parameter to move the resulting torch tensors to the specified device

    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def to_numpy(self, x: torch.Tensor) -> NDArray:
        """
        Convert a detached [torch.Tensor][torch.Tensor] to [numpy.ndarray][numpy.ndarray]
        """
        arr: NDArray = x.cpu().numpy()
        return arr

    def from_numpy(self, x: NDArray) -> torch.Tensor:
        """
        Convert a [numpy.ndarray][numpy.ndarray] to [torch.Tensor][torch.Tensor] and optionally move it to a provided
        device
        """
        t = torch.from_numpy(x)
        if self.device is not None:
            t = t.to(self.device)
        return t


class TorchCatAggregator(TensorAggregator):
    """
    An aggregator that concatenates tensors using PyTorch's [torch.cat][torch.cat] function. This class
    is designed to aggregate tensors from generators, either nested or single-level, into a
    single tensor by concatenating them along a specified dimension.
    """

    def aggregate_from_nested_generators(
        self,
        nested_generators_of_tensors: Generator[
            Generator[torch.Tensor, None, None], None, None
        ],
    ) -> torch.Tensor:
        """
        Aggregates tensors from a nested generator structure into a single tensor by concatenating.
        Each inner generator is first concatenated along dimension 1 into a tensor, and then
        these tensors are concatenated along dimension 0 together to form the final tensor.

        Args:
            nested_generators_of_tensors: A generator of generators, where each inner generator
                yields `torch.Tensor` objects.

        Returns:
            A single tensor formed by concatenating all tensors from the nested generators.

        """
        return torch.cat(
            list(
                map(
                    lambda tensor_gen: torch.cat(list(tensor_gen), dim=1),
                    nested_generators_of_tensors,
                )
            )
        )

    def aggregate_from_generator(
        self, tensor_generator: Generator[torch.Tensor, None, None]
    ) -> torch.Tensor:
        """
        Aggregates tensors from a single-level generator into a single tensor by concatenating them.
        This method is a straightforward way to combine a sequence of tensors into one larger tensor.

        Args:
            tensor_generator: A generator that yields `torch.Tensor` objects.

        Returns:
            torch.Tensor: A single tensor formed by concatenating all tensors from the generator.
                          The concatenation is performed along the default dimension (0).
        """
        return torch.cat(list(tensor_generator))
