"""
This module provides classes and utilities for handling large arrays that are chunked
and lazily evaluated. It includes abstract base classes for converting between tensor
types and NumPy arrays, aggregating blocks of data, and abstract representations of
lazy arrays. Concrete implementations are provided for handling chunked lazy arrays
(chunked in one resp. two dimensions), with support for efficient storage and retrieval
using the Zarr library.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import zarr
from numpy.typing import NDArray
from tqdm import tqdm
from zarr.storage import StoreLike

from ..utils import log_duration
from .types import TensorType


class NumpyConverter(Generic[TensorType], ABC):
    """
    Base class for converting TensorType objects into numpy arrays and vice versa.
    """

    @abstractmethod
    def to_numpy(self, x: TensorType) -> NDArray:
        """Override this method for converting a TensorType object into a numpy array"""

    @abstractmethod
    def from_numpy(self, x: NDArray) -> TensorType:
        """Override this method for converting a numpy array into a TensorType object"""


class SequenceAggregator(Generic[TensorType], ABC):
    @abstractmethod
    def __call__(
        self,
        tensor_sequence: LazyChunkSequence,
    ):
        """
        Aggregates tensors from a sequence.

        Implement this method to define how a sequence of tensors, provided by a
        generator, should be combined.
        """


class ListAggregator(SequenceAggregator):
    def __call__(
        self,
        tensor_sequence: LazyChunkSequence,
    ) -> List[TensorType]:
        """
        Aggregates tensors from a single-level generator into a list. This method simply
        collects each tensor emitted by the generator into a single list.

        Args:
            tensor_sequence: Object wrapping a generator that yields `TensorType`
                objects.

        Returns:
            A list containing all the tensors provided by the tensor_generator.
        """

        gen = cast(Iterator[TensorType], tensor_sequence.generator_factory())

        if tensor_sequence.len_generator is not None:
            gen = cast(
                Iterator[TensorType],
                tqdm(gen, total=tensor_sequence.len_generator, desc="Blocks"),
            )

        return [t for t in gen]


class NestedSequenceAggregator(Generic[TensorType], ABC):
    @abstractmethod
    def __call__(self, nested_sequence_of_tensors: NestedLazyChunkSequence):
        """
        Aggregates tensors from a nested sequence of tensors.

        Implement this method to specify how tensors, nested in two layers of
        generators, should be combined. Useful for complex data structures where tensors
        are not directly accessible in a flat list.
        """


class NestedListAggregator(NestedSequenceAggregator):
    def __call__(
        self,
        nested_sequence_of_tensors: NestedLazyChunkSequence,
    ) -> List[List[TensorType]]:
        """
         Aggregates tensors from a nested generator structure into a list of lists.
         Each inner generator is converted into a list of tensors, resulting in a nested
         list structure.

         Args:
             nested_sequence_of_tensors: Object wrapping a generator of generators,
                where each inner generator yields TensorType objects.

        Returns:
            A list of lists, where each inner list contains tensors returned from one
                of the inner generators.
        """
        outer_gen = cast(
            Iterator[Iterator[TensorType]],
            nested_sequence_of_tensors.generator_factory(),
        )
        len_outer_gen = nested_sequence_of_tensors.len_outer_generator
        if len_outer_gen is not None:
            outer_gen = cast(
                Iterator[Iterator[TensorType]],
                tqdm(outer_gen, total=len_outer_gen, desc="Row blocks"),
            )

        return [list(tensor_gen) for tensor_gen in outer_gen]


class LazyChunkSequence(Generic[TensorType]):
    """
    A class representing a chunked, and lazily evaluated array,
    where the chunking is restricted to the first dimension

    This class is designed to handle large arrays that don't fit in memory.
    It works by generating chunks of the array on demand and can
    also convert these chunks to a Zarr array
    for efficient storage and retrieval.

    Attributes:
        generator_factory: A factory function that returns
            a generator. This generator yields chunks of the large array when called.
        len_generator: if the number of elements from the generator is
            known from the context, this optional parameter can be used to improve
            logging by adding a progressbar.
    """

    def __init__(
        self,
        generator_factory: Callable[[], Generator[TensorType, None, None]],
        len_generator: Optional[int] = None,
    ):
        self.generator_factory = generator_factory
        self.len_generator = len_generator

    @log_duration(log_level=logging.INFO)
    def compute(self, aggregator: Optional[SequenceAggregator] = None) -> Any:
        """
        Computes and optionally aggregates the chunks of the array using the provided
        aggregator. This method initiates the generation of chunks and then
        combines them according to the aggregator's logic.

        Args:
            aggregator: An optional aggregator for combining the chunks of
                the array. If None, a default ListAggregator is used to simply collect
                the chunks into a list.

        Returns:
            The aggregated result of all chunks of the array, the format of which
                depends on the aggregator used.

        """
        if aggregator is None:
            aggregator = ListAggregator()
        return aggregator(self)

    @log_duration(log_level=logging.INFO)
    def to_zarr(
        self,
        path_or_url: Union[str, StoreLike],
        converter: NumpyConverter,
        return_stored: bool = False,
        overwrite: bool = False,
    ) -> Optional[zarr.Array]:
        """
        Converts the array into Zarr format, a storage format optimized for large
        arrays, and stores it at the specified path or URL. This method is suitable for
        scenarios where the data needs to be saved for later use or for large datasets
        requiring efficient storage.

        Args:
            path_or_url: The file path or URL where the Zarr array will be stored.
                Also excepts instances of zarr stores.
            converter: A converter for transforming blocks into NumPy arrays
                compatible with Zarr.
            return_stored: If True, the method returns the stored Zarr array; otherwise,
                it returns None.
            overwrite: If True, overwrites existing data at the given path_or_url.
                If False, an error is raised in case of existing data.

        Returns:
            The Zarr array if return_stored is True; otherwise, None.
        """
        row_idx = 0
        z = None

        gen = cast(Iterator[TensorType], self.generator_factory())

        if self.len_generator is not None:
            gen = cast(
                Iterator[TensorType], tqdm(gen, total=self.len_generator, desc="Blocks")
            )

        for block in gen:
            numpy_block = converter.to_numpy(block)

            if z is None:
                z = self._initialize_zarr_array(numpy_block, path_or_url, overwrite)

            new_shape = self._new_shape_according_to_block(numpy_block, row_idx)
            z.resize(new_shape)

            z[row_idx : row_idx + numpy_block.shape[0]] = numpy_block
            row_idx += numpy_block.shape[0]

        return z if return_stored else None

    @staticmethod
    def _new_shape_according_to_block(
        block: NDArray,
        current_row_idx: int,
    ):
        return (current_row_idx + block.shape[0],) + block.shape[1:]

    @staticmethod
    def _initialize_zarr_array(block: NDArray, path_or_url: str, overwrite: bool):
        initial_shape = (0,) + block.shape[1:]
        return zarr.open(
            path_or_url,
            mode="w" if overwrite else "w-",
            shape=initial_shape,
            chunks=block.shape,
            dtype=block.dtype,
        )


class NestedLazyChunkSequence(Generic[TensorType]):
    """
    A class representing chunked, and lazily evaluated array, where the chunking is
    restricted to the first two dimensions.

    This class is designed for handling large arrays where individual chunks are
    loaded and processed lazily. It supports converting these chunks into a Zarr array
    for efficient storage and retrieval, with chunking applied along the first two
    dimensions.

    Attributes:
        generator_factory: A factory function that returns a generator of generators.
            Each inner generator yields chunks
        len_outer_generator: if the number of elements from the outer generator is
            known from the context, this optional parameter can be used to improve
            logging by adding a progressbar.
    """

    def __init__(
        self,
        generator_factory: Callable[
            [], Generator[Generator[TensorType, None, None], None, None]
        ],
        len_outer_generator: Optional[int] = None,
    ):
        self.generator_factory = generator_factory
        self.len_outer_generator = len_outer_generator

    @log_duration(log_level=logging.INFO)
    def compute(self, aggregator: Optional[NestedSequenceAggregator] = None) -> Any:
        """
        Computes and optionally aggregates the chunks of the array using the provided
        aggregator. This method initiates the generation of chunks and then
        combines them according to the aggregator's logic.

        Args:
            aggregator: An optional aggregator for combining the chunks of
                the array. If None, a default
                [NestedListAggregator][pydvl.influence.array.NestedListAggregator]
                is used to simply collect the chunks into a list of lists.

        Returns:
            The aggregated result of all chunks of the array, the format of which
            depends on the aggregator used.

        """
        if aggregator is None:
            aggregator = NestedListAggregator()
        return aggregator(self)

    @log_duration(log_level=logging.INFO)
    def to_zarr(
        self,
        path_or_url: Union[str, StoreLike],
        converter: NumpyConverter,
        return_stored: bool = False,
        overwrite: bool = False,
    ) -> Optional[zarr.Array]:
        """
        Converts the array into Zarr format, a storage format optimized for large
        arrays, and stores it at the specified path or URL. This method is suitable for
        scenarios where the data needs to be saved for later use or for large datasets
        requiring efficient storage.

        Args:
            path_or_url: The file path or URL where the Zarr array will be stored.
                Also excepts instances of zarr stores.
            converter: A converter for transforming blocks into NumPy arrays
                compatible with Zarr.
            return_stored: If True, the method returns the stored Zarr array;
                otherwise, it returns None.
            overwrite: If True, overwrites existing data at the given path_or_url.
                If False, an error is raised in case of existing data.

        Returns:
            The Zarr array if return_stored is True; otherwise, None.
        """

        row_idx = 0
        z = None
        numpy_block = None
        block_generator = cast(Iterator[Iterator[TensorType]], self.generator_factory())

        if self.len_outer_generator is not None:
            block_generator = cast(
                Iterator[Iterator[TensorType]],
                tqdm(
                    block_generator, total=self.len_outer_generator, desc="Row blocks"
                ),
            )

        for row_blocks in block_generator:
            col_idx = 0
            for block in row_blocks:
                numpy_block = converter.to_numpy(block)
                if z is None:
                    z = self._initialize_zarr_array(numpy_block, path_or_url, overwrite)
                new_shape = self._new_shape_according_to_block(
                    z, numpy_block, row_idx, col_idx
                )
                z.resize(new_shape)
                idx_slice_to_update = self._idx_slice_for_update(
                    numpy_block, row_idx, col_idx
                )
                z[idx_slice_to_update] = numpy_block

                col_idx += numpy_block.shape[1]

            if numpy_block is None:
                raise ValueError("Generator is empty")

            row_idx += numpy_block.shape[0]

        return z if return_stored else None

    @staticmethod
    def _idx_slice_for_update(
        block: NDArray, current_row_idx: int, current_col_idx
    ) -> Tuple[int, ...]:
        idx = [
            slice(current_row_idx, current_row_idx + block.shape[0]),
            slice(current_col_idx, current_col_idx + block.shape[1]),
        ]
        idx.extend(slice(s) for s in block.shape[2:])
        return tuple(idx)  # type:ignore

    @staticmethod
    def _new_shape_according_to_block(
        zarr_array: zarr.Array,
        block: NDArray,
        current_row_idx: int,
        current_col_idx: int,
    ):
        return (
            max(current_row_idx + block.shape[0], zarr_array.shape[0]),
            max(current_col_idx + block.shape[1], zarr_array.shape[1]),
        ) + block.shape[2:]

    @staticmethod
    def _initialize_zarr_array(
        block: NDArray, path_or_url: str, overwrite: bool
    ) -> zarr.Array:
        fixed_shape = block.shape[2:]
        initial_shape = (0, 0) + fixed_shape
        chunk_size = block.shape[:2] + fixed_shape
        return zarr.open(
            path_or_url,
            mode="w" if overwrite else "w-",
            shape=initial_shape,
            chunks=chunk_size,
            dtype=block.dtype,
        )
