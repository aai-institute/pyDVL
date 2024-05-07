import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import (
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import dask
import numpy as np
import torch
from dask import array as da
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm import tqdm

from ...utils.exceptions import catch_and_raise_exception
from ..array import (
    LazyChunkSequence,
    NestedLazyChunkSequence,
    NestedSequenceAggregator,
    NumpyConverter,
    SequenceAggregator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "to_model_device",
    "TorchTensorContainerType",
    "align_structure",
    "align_with_model",
    "flatten_dimensions",
    "TorchNumpyConverter",
    "TorchCatAggregator",
    "NestedTorchCatAggregator",
    "torch_dataset_to_dask_array",
    "EkfacRepresentation",
    "empirical_cross_entropy_loss_fn",
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

    This function takes a 1D tensor (input_vector) and reshapes it into a series of
    tensors with shapes given by 'target_shapes'.
    The reshaped tensors are returned as a tuple in the same order
    as their corresponding shapes.

    Note:
        The total number of elements in 'input_vector' must be equal to the
            sum of the products of the shapes in 'target_shapes'.

    Args:
        input_vector: The 1D tensor to be reshaped. Must be 1D.
        target_shapes: An iterable of tuples. Each tuple defines the shape of a tensor
            to be reshaped from the 'input_vector'.

    Returns:
        A tuple of reshaped tensors.

    Raises:
        ValueError: If 'input_vector' is not a 1D tensor or if the total
            number of elements in 'input_vector' does not
            match the sum of the products of the shapes in 'target_shapes'.
    """

    if input_vector.dim() != 1:
        raise ValueError("Input vector must be a 1D tensor")

    total_elements = sum(math.prod(shape) for shape in target_shapes)

    if total_elements != input_vector.shape[0]:
        raise ValueError(
            f"The total elements in shapes {total_elements} "
            f"does not match the vector length {input_vector.shape[0]}"
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
                "The shapes of the values in 'target' do not match the shapes "
                "of the values in 'source'."
            )

        tangent_dict = target

    elif isinstance(target, tuple) or isinstance(target, list):
        if [v.shape for v in target] != [v.shape for v in source.values()]:
            raise ValueError(
                "'target' is a tuple/list but its elements' shapes do not match "
                "the shapes of the values in 'source'."
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
                f"'target' is a tensor but cannot be reshaped to match 'source'. "
                f"Original error: {e}"
            )

    else:
        raise ValueError(f"'target' is of type {type(target)} which is not supported.")

    return tangent_dict


def align_with_model(x: TorchTensorContainerType, model: torch.nn.Module):
    """
    Aligns an input to the model's parameter structure, i.e. transforms it into a dict
    with the same keys as model.named_parameters() and matching tensor shapes

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
    Flattens the dimensions of each tensor in the given iterable and concatenates them
    along a specified dimension.

    This function takes an iterable of PyTorch tensors and flattens each tensor.
    Optionally, each tensor can be reshaped to a specified shape before concatenation.
    The concatenation is performed along the dimension specified by `concat_at`.

    Args:
        tensors: An iterable containing PyTorch tensors to be flattened
            and concatenated.
        shape: A tuple representing the desired shape to which each tensor is reshaped
            before concatenation. If None, tensors are flattened to 1D.
        concat_at: The dimension along which to concatenate the tensors.

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
    resulting_dtype: Type[np.number] = np.float32,
) -> Tuple[da.Array, ...]:
    """
    Construct tuple of dask arrays from a PyTorch dataset, using dask.delayed

    Args:
        dataset: A PyTorch [dataset][torch.utils.data.Dataset]
        chunk_size: The size of the chunks for the resulting Dask arrays.
        total_size: If the dataset does not implement len, provide the length
            via this parameter. If None
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
                    f"The number of samples in the dataset ({n_data}), derived "
                    f"from calling ´len´, does not match the provided "
                    f"total number of samples ({total_size}). "
                    f"Call the function without total_size."
                )
            return n_data
        except TypeError as e:
            err_msg = (
                f"Could not infer the number of samples in the dataset from "
                f"calling ´len´. Original error: {e}."
            )
            if total_size is not None:
                logger.warning(
                    err_msg
                    + f" Using the provided total number of samples {total_size}."
                )
                return total_size
            else:
                logger.warning(
                    err_msg + " Infer the number of samples from the dataset, "
                    "via iterating the dataset once. "
                    "This might induce severe overhead, so consider"
                    "providing total_size, if you know the number of samples "
                    "beforehand."
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
    Helper class for converting between [torch.Tensor][torch.Tensor] and
    [numpy.ndarray][numpy.ndarray]

    Args:
        device: Optional device parameter to move the resulting torch tensors to the
            specified device

    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def to_numpy(self, x: torch.Tensor) -> NDArray:
        """
        Convert a detached [torch.Tensor][torch.Tensor] to
        [numpy.ndarray][numpy.ndarray]
        """
        arr: NDArray = x.cpu().numpy()
        return arr

    def from_numpy(self, x: NDArray) -> torch.Tensor:
        """
        Convert a [numpy.ndarray][numpy.ndarray] to [torch.Tensor][torch.Tensor] and
        optionally move it to a provided device
        """
        t = torch.from_numpy(x)
        if self.device is not None:
            t = t.to(self.device)
        return t


class TorchCatAggregator(SequenceAggregator[torch.Tensor]):
    """
    An aggregator that concatenates tensors using PyTorch's [torch.cat][torch.cat]
    function. Concatenation is done along the first dimension of the chunks.
    """

    def __call__(
        self,
        tensor_sequence: LazyChunkSequence[torch.Tensor],
    ):
        """
        Aggregates tensors from a single-level generator into a single tensor by
        concatenating them. This method is a straightforward way to combine a sequence
        of tensors into one larger tensor.

        Args:
            tensor_sequence: Object wrapping a generator that yields `torch.Tensor`
                objects.

        Returns:
            A single tensor formed by concatenating all tensors from the generator.
                The concatenation is performed along the default dimension (0).
        """
        t_gen = cast(Iterator[torch.Tensor], tensor_sequence.generator_factory())
        len_generator = tensor_sequence.len_generator
        if len_generator is not None:
            t_gen = cast(
                Iterator[torch.Tensor], tqdm(t_gen, total=len_generator, desc="Blocks")
            )

        return torch.cat(list(t_gen))


class NestedTorchCatAggregator(NestedSequenceAggregator[torch.Tensor]):
    """
    An aggregator that concatenates tensors using PyTorch's [torch.cat][torch.cat]
    function. Concatenation is done along the first two dimensions of the chunks.
    """

    def __call__(
        self, nested_sequence_of_tensors: NestedLazyChunkSequence[torch.Tensor]
    ):
        """
        Aggregates tensors from a nested generator structure into a single tensor by
        concatenating. Each inner generator is first concatenated along dimension 1 into
        a tensor, and then these tensors are concatenated along dimension 0 together to
        form the final tensor.

        Args:
            nested_sequence_of_tensors: Object wrapping a generator of generators,
                where each inner generator yields `torch.Tensor` objects.

        Returns:
            A single tensor formed by concatenating all tensors from the nested
            generators.

        """

        outer_gen = cast(
            Iterator[Iterator[torch.Tensor]],
            nested_sequence_of_tensors.generator_factory(),
        )
        len_outer_generator = nested_sequence_of_tensors.len_outer_generator
        if len_outer_generator is not None:
            outer_gen = cast(
                Iterator[Iterator[torch.Tensor]],
                tqdm(outer_gen, total=len_outer_generator, desc="Row blocks"),
            )

        return torch.cat(
            list(
                map(
                    lambda tensor_gen: torch.cat(list(tensor_gen), dim=1),
                    outer_gen,
                )
            )
        )


@dataclass(frozen=True)
class EkfacRepresentation:
    r"""
    Container class for the EKFAC representation of the Hessian.
    It can be iterated over to get the layers names and their corresponding module,
    eigenvectors and diagonal elements of the factorized Hessian matrix.

    Args:
        layer_names: Names of the layers.
        layers_module: The layers.
        evecs_a: The a eigenvectors of the ekfac representation.
        evecs_g: The g eigenvectors of the ekfac representation.
        diags: The diagonal elements of the factorized Hessian matrix.
    """
    layer_names: Iterable[str]
    layers_module: Iterable[torch.nn.Module]
    evecs_a: Iterable[torch.Tensor]
    evecs_g: Iterable[torch.Tensor]
    diags: Iterable[torch.Tensor]

    def __iter__(self):
        return iter(
            zip(
                self.layer_names,
                zip(self.layers_module, self.evecs_a, self.evecs_g, self.diags),
            )
        )

    def get_layer_evecs(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        It returns two dictionaries, one for the a eigenvectors and one for the g
        eigenvectors, with the layer names as keys. The eigenvectors are in the same
        order as the layers in the model.
        """
        evecs_a_dict = {layer_name: evec_a for layer_name, (_, evec_a, _, _) in self}
        evecs_g_dict = {layer_name: evec_g for layer_name, (_, _, evec_g, _) in self}
        return evecs_a_dict, evecs_g_dict

    def to(self, device: torch.device) -> "EkfacRepresentation":
        return EkfacRepresentation(
            self.layer_names,
            [layer.to(device) for layer in self.layers_module],
            [evec_a.to(device) for evec_a in self.evecs_a],
            [evec_g.to(device) for evec_g in self.evecs_g],
            [diag.to(device) for diag in self.diags],
        )


def empirical_cross_entropy_loss_fn(
    model_output: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Computes the empirical cross entropy loss of the model output. This is the
    cross entropy loss of the model output without the labels. The function takes
    all the usual arguments and keyword arguments of the cross entropy loss
    function, so that it is compatible with the PyTorch cross entropy loss
    function. However, it ignores everything except the first argument, which is
    the model output.

    Args:
        model_output: The output of the model.
    """
    probs_ = torch.softmax(model_output, dim=1)
    log_probs_ = torch.log(probs_)
    log_probs_ = torch.where(
        torch.isfinite(log_probs_), log_probs_, torch.zeros_like(log_probs_)
    )
    return torch.sum(log_probs_ * probs_.detach() ** 0.5)


@catch_and_raise_exception(RuntimeError, lambda e: TorchLinalgEighException(e))
def safe_torch_linalg_eigh(*args, **kwargs):
    """
    A wrapper around `torch.linalg.eigh` that safely handles potential runtime errors
    by raising a custom `TorchLinalgEighException` with more context,
    especially related to the issues reported in
    [https://github.com/pytorch/pytorch/issues/92141](
    https://github.com/pytorch/pytorch/issues/92141).

    Args:
        *args: Positional arguments passed to `torch.linalg.eigh`.
        **kwargs: Keyword arguments passed to `torch.linalg.eigh`.

    Returns:
        The result of calling `torch.linalg.eigh` with the provided arguments.

    Raises:
        TorchLinalgEighException: If a `RuntimeError` occurs during the execution of
            `torch.linalg.eigh`.
    """
    return torch.linalg.eigh(*args, **kwargs)


class TorchLinalgEighException(Exception):
    """
    Exception to wrap a RunTimeError raised by torch.linalg.eigh, when used
    with large matrices,
    see [https://github.com/pytorch/pytorch/issues/92141](
    https://github.com/pytorch/pytorch/issues/92141)
    """

    def __init__(self, original_exception: RuntimeError):
        func = torch.linalg.eigh
        err_msg = (
            f"A RunTimeError occurred in '{func.__module__}.{func.__qualname__}'. "
            "This might be related to known issues with "
            "[torch.linalg.eigh][torch.linalg.eigh] on certain matrix sizes.\n "
            "For more details, refer to "
            "https://github.com/pytorch/pytorch/issues/92141. \n"
            "In this case, consider to use a different implementation, which does not "
            "depend on the usage of [torch.linalg.eigh][torch.linalg.eigh].\n"
            f" Inspect the original exception message: \n{str(original_exception)}"
        )
        super().__init__(err_msg)
