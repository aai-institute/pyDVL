"""
This module provides classes and utilities for handling large arrays that are chunked and lazily evaluated.
It includes abstract base classes for converting between tensor types and NumPy arrays, aggregating blocks
of data, and abstract representations of lazy arrays. Concrete implementations are provided for handling
chunked lazy arrays (chunked in one resp. two dimensions), with support for efficient storage and retrieval
using the Zarr library.

Classes:
    NumpyConverter: Abstract base class for converting between tensor types and NumPy arrays.
    BlockAggregator: Abstract base class for aggregating blocks of data.
    ListAggregator: Concrete implementation of BlockAggregator using lists.
    LazyArray: Abstract base class representing a lazily evaluated array.
    OneDimChunkedLazyArray: Represents a chunked and lazily evaluated array, where chunking is restricted to the
        first dimension.
    TwoDimChunkedLazyArray: Represents a chunked and lazily evaluated array, where chunking is restricted to the
        two first dimension.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generator, Generic, Optional, Tuple

import zarr
from numpy.typing import NDArray

from .base_influence_model import TensorType


class NumpyConverter(Generic[TensorType], ABC):
    """
    Base class for converting TensorType objects into numpy arrays and vice versa.
    """

    @abstractmethod
    def to_numpy(self, x: TensorType) -> NDArray:
        """Overwrite this method for converting a TensorType object into a numpy array"""

    @abstractmethod
    def from_numpy(self, x: NDArray) -> TensorType:
        """Overwrite this method for converting a numpy array into a TensorType object"""


class BlockAggregator(Generic[TensorType], ABC):
    @abstractmethod
    def aggregate_nested(
        self, tensors: Generator[Generator[TensorType, None, None], None, None]
    ):
        """Overwrite this method to aggregate provided blocks into a single tensor"""

    @abstractmethod
    def aggregate(self, tensors: Generator[TensorType, None, None]):
        """Overwrite this method to aggregate provided list of tensors into a single tensor"""


class ListAggregator(BlockAggregator):
    def aggregate_nested(
        self, tensors: Generator[Generator[TensorType, None, None], None, None]
    ):
        return [list(tensor_gen) for tensor_gen in tensors]

    def aggregate(self, tensors: Generator[TensorType, None, None]):
        return [t for t in tensors]


class LazyArray(ABC):
    @abstractmethod
    def compute(self, block_aggregator: Optional[BlockAggregator] = None):
        pass

    @abstractmethod
    def to_zarr(
        self,
        path_or_url: str,
        numpy_converter: NumpyConverter,
        return_stored: bool = False,
        overwrite: bool = False,
    ):
        pass


class OneDimChunkedLazyArray(LazyArray):
    """
    A class representing chunked, and lazily evaluated array, where the chunking is restricted to the first dimension

    This class is designed to handle large arrays that are not feasible to load into memory all at once.
    It works by generating chunks of the array on demand and can also convert these chunks to a Zarr array
    for efficient storage and retrieval.

    Attributes:
        generator_factory: A factory function that returns
            a generator. This generator yields chunks of the large array when called.
    """

    def __init__(
        self, generator_factory: Callable[[], Generator[TensorType, None, None]]
    ):
        self.generator_factory = generator_factory

    def compute(self, block_aggregator: Optional[BlockAggregator] = None):
        """
        Computes and optionally aggregates the chunks of the array.

        Args:
            block_aggregator (Optional[BlockAggregator]): An optional aggregator for combining the chunks
                of the array. If None, a default ListAggregator is used.

        Returns:
            The aggregated result of all chunks of the array.
        """
        if block_aggregator is None:
            block_aggregator = ListAggregator()
        return block_aggregator.aggregate(self.generator_factory())

    def to_zarr(
        self,
        path_or_url: str,
        numpy_converter: NumpyConverter,
        return_stored: bool = False,
        overwrite: bool = False,
    ):
        """
        Converts the array into a Zarr format and stores it at the specified location.

        Args:
            path_or_url: The file path or URL where the Zarr array will be stored.
            numpy_converter: A converter to transform blocks into NumPy arrays.
            return_stored: If True, returns the stored Zarr array. Defaults to False.
            overwrite: If True, overwrites the existing data at the path_or_url, otherwise raises an error.
                Defaults to False.

        Returns:
            The Zarr array if return_stored is True; otherwise, None.
        """
        row_idx = 0
        z = None
        for block in self.generator_factory():
            numpy_block = numpy_converter.to_numpy(block)

            if z is None:
                z = self._initialize_zarr_array(numpy_block, path_or_url, overwrite)

            new_shape = self._new_shape_according_to_block(numpy_block, row_idx)
            z.resize(new_shape)

            z[row_idx : row_idx + numpy_block.shape[0]] = numpy_block
            row_idx += numpy_block.shape[0]

        if return_stored:
            return z

    @staticmethod
    def _new_shape_according_to_block(
        block: NDArray,
        current_row_idx: int,
    ) -> Tuple[int, ...]:
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


class TwoDimChunkedLazyArray(LazyArray):
    """
    A class representing chunked, and lazily evaluated array, where the chunking is restricted to the
    first two dimensions.

    This class is designed for handling large arrays where individual chunks are
    loaded and processed lazily. It supports converting these chunks into a Zarr array for efficient
    storage and retrieval, with chunking applied along the first two dimensions.

    Attributes:
        generator_factory (Callable[[], Generator[Generator[NDArray, None, None], None, None]]):
            A factory function that returns a generator of generators. Each inner generator yields
            chunks.
    """

    def __init__(
        self,
        generator_factory: Callable[
            [], Generator[Generator[TensorType, None, None], None, None]
        ],
    ):
        self.generator_factory = generator_factory

    def compute(self, block_aggregator: Optional[BlockAggregator] = None):
        """
        Computes and optionally aggregates the chunks of the array.

        Args:
            block_aggregator: An optional aggregator for combining the chunks
                of the array. If None, a default ListAggregator is used.

        Returns:
            The aggregated result of all chunks of the two-dimensional array.
        """
        if block_aggregator is None:
            block_aggregator = ListAggregator()
        return block_aggregator.aggregate_nested(self.generator_factory())

    def to_zarr(
        self,
        path_or_url: str,
        numpy_converter: NumpyConverter,
        return_stored: bool = False,
        overwrite: bool = False,
    ):
        """
        Converts the array into a Zarr format and stores it at the specified location.

        Args:
            path_or_url: The file path or URL where the Zarr array will be stored.
            numpy_converter: A converter to transform blocks into NumPy arrays.
            return_stored: If True, returns the stored Zarr array. Defaults to False.
            overwrite: If True, overwrites the existing data at path_or_url. Defaults to False.

        Returns:
            The Zarr array if return_stored is True; otherwise, None.
        """

        row_idx = 0
        z = None
        numpy_block = None
        for row_blocks in self.generator_factory():
            col_idx = 0
            for block in row_blocks:
                numpy_block = numpy_converter.to_numpy(block)
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

        if return_stored:
            return z

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
    ) -> Tuple[int, ...]:
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
            overwrite=overwrite,
        )
