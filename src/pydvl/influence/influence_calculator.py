"""
This module provides functionality for calculating influences for large amount of data.
The computation is based on a chunk computation model in the form of an instance of
[InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel],
which is mapped over collection of chunks.
"""

import logging
from functools import partial
from typing import Generator, Iterable, Optional, Sized, Tuple, Type, Union, cast

import distributed
from dask import array as da
from dask import delayed
from distributed import Client
from numpy.typing import NDArray

from .array import LazyChunkSequence, NestedLazyChunkSequence, NumpyConverter
from .base_influence_function_model import InfluenceFunctionModel
from .types import InfluenceMode, TensorType, UnsupportedInfluenceModeException

__all__ = [
    "DaskInfluenceCalculator",
    "SequentialInfluenceCalculator",
    "DisableClientSingleThreadCheck",
]

logger = logging.getLogger(__name__)


class InvalidDimensionChunksError(ValueError):
    def __init__(self, chunks: Tuple[Tuple[int, ...], ...]):
        msg = (
            f"Array must be un-chunked in every dimension but the first, got {chunks=}"
        )
        super().__init__(msg)


class UnalignedChunksError(ValueError):
    def __init__(self, chunk_sizes_x: Tuple[int, ...], chunk_sizes_y: Tuple[int, ...]):
        msg = (
            f"Arrays x and y must have the same chunking in the first dimension, got "
            f"{chunk_sizes_x=} and {chunk_sizes_y=}"
        )
        super().__init__(msg)


class ThreadSafetyViolationError(ValueError):
    def __init__(self):
        msg = (
            "The operation cannot proceed because the "
            "influence_function_model is not thread-safe, yet it is being used in a "
            "context that requires thread safety. Please ensure that all components "
            "are compatible with multi-threaded environments or use single-threaded "
            "workers. \n"
            "client = Client(threads_per_worker=1)"
        )
        super().__init__(msg)


class DisableClientSingleThreadCheck:
    """
    This type can be provided to the initialization of a DaskInfluenceCalculator
    instead of a distributed client object. It is useful in those scenarios, where
    the user want to disable the checking for thread-safety in the initialization
    phase, e.g. when using the single machine synchronous scheduler for debugging
    purposes.

    Example:
        ```python
        from pydvl.influence import DisableClientThreadingCheck

        da_calc = DaskInfluenceCalculator(if_model,
                                          TorchNumpyConverter(),
                                          DisableClientThreadingCheck)
        da_influences = da_calc.influences(da_x_test, da_y_test, da_x, da_y)
        da_influences.compute(scheduler='synchronous')
        ```
    """

    @staticmethod
    def warning_msg() -> str:
        msg = (
            "You have disabled client threading safety check. "
            "This is only advised for debugging "
            "purposes with the single machine synchronous scheduler. "
            "Disabling these checks can lead to unpredictable behavior and "
            "is only advised when you are fully aware of the implications."
        )
        return msg


class DaskInfluenceCalculator:
    """
    This class is designed to compute influences over
    [dask.array.Array][dask.array.Array] collections, leveraging the capabilities of
    Dask for distributed computing and parallel processing. It requires an influence
    computation model of type
    [InfluenceFunctionModel]
    [pydvl.influence.base_influence_function_model.InfluenceFunctionModel],
    which defines how influences are computed on a chunk of data.
    Essentially, this class functions by mapping the influence
    function model across the various chunks of a [dask.array.Array][dask.array.Array]
    collection.

    Args:
        influence_function_model: instance of type
            [InfluenceFunctionModel]
            [pydvl.influence.base_influence_function_model.InfluenceFunctionModel], that
            specifies the computation logic for influence on data chunks. It's a pivotal
            part of the calculator, determining how influence is computed and applied
            across the data array.
        converter: A utility for converting numpy arrays to TensorType objects,
            facilitating the interaction between numpy arrays and the influence
            function model.
        client: This parameter accepts either of two types:

            1. A distributed [Client][distributed.Client] object

            2. The special type [DisableClientSingleThreadCheck]
            [pydvl.influence.influence_calculator.DisableClientSingleThreadCheck], which
            serves as a flag to bypass certain checks.

            During initialization, the system verifies if all workers are operating in
            single-threaded mode when the provided influence_function_model is
            designated as not thread-safe (indicated by the `is_thread_safe` property).
            If this condition is not met, the initialization will raise a specific
            error, signaling a potential thread-safety conflict.

            To intentionally skip this safety check
            (e.g., for debugging purposes using the single machine synchronous
            scheduler), you can supply the [DisableClientSingleThreadCheck]
            [pydvl.influence.influence_calculator.DisableClientSingleThreadCheck] type.

    !!! Warning
        Make sure to set `threads_per_worker=1`, when using the distributed scheduler
        for computing, if your implementation of
        [InfluenceFunctionModel]
        [pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
        is not thread-safe.
        ```python
        client = Client(threads_per_worker=1)
        ```
        For details on dask schedulers see the
        [official documentation](https://docs.dask.org/en/stable/scheduling.html).

    Example:
        ```python
        import torch
        from torch.utils.data import Dataset, DataLoader
        from pydvl.influence import DaskInfluenceCalculator
        from pydvl.influence.torch import CgInfluence
        from pydvl.influence.torch.util import (
            torch_dataset_to_dask_array,
            TorchNumpyConverter,
        )
        from distributed import Client

        # Possible some out of memory large Dataset
        train_data_set: Dataset = LargeDataSet(...)
        test_data_set: Dataset = LargeDataSet(...)

        train_dataloader = DataLoader(train_data_set)
        infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
        infl_model = if_model.fit(train_dataloader)

        # wrap your input data into dask arrays
        chunk_size = 10
        da_x, da_y = torch_dataset_to_dask_array(train_data_set, chunk_size=chunk_size)
        da_x_test, da_y_test = torch_dataset_to_dask_array(test_data_set,
                                                           chunk_size=chunk_size)

        # use only one thread for scheduling, due to non-thread safety of some torch
        # operations
        client = Client(n_workers=4, threads_per_worker=1)

        infl_calc = DaskInfluenceCalculator(infl_model,
                                            TorchNumpyConverter(device=torch.device("cpu")),
                                            client)
        da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
        # da_influences is a dask.array.Array

        # trigger computation and write chunks to disk in parallel
        da_influences.to_zarr("path/or/url")
        ```

    """

    def __init__(
        self,
        influence_function_model: InfluenceFunctionModel,
        converter: NumpyConverter,
        client: Union[Client, Type[DisableClientSingleThreadCheck]],
    ):
        self._n_parameters = influence_function_model.n_parameters
        self.influence_function_model = influence_function_model
        self.numpy_converter = converter

        if isinstance(client, type(DisableClientSingleThreadCheck)):
            logger.warning(DisableClientSingleThreadCheck.warning_msg())
            self.influence_function_model = delayed(influence_function_model)
        elif isinstance(client, Client):
            self._validate_client(client, influence_function_model)
            self.influence_function_model = client.scatter(
                influence_function_model, broadcast=True
            )
        else:
            raise ValueError(
                "The 'client' parameter "
                "must either be a distributed.Client object or the"
                "type 'DisableClientSingleThreadCheck'."
            )

    @staticmethod
    def _validate_client(
        client: Client, influence_function_model: InfluenceFunctionModel
    ):
        info = client.scheduler_info()
        workers = info["workers"]
        all_single_threaded = all(
            worker["nthreads"] == 1 for worker in workers.values()
        )
        if not (influence_function_model.is_thread_safe or all_single_threaded):
            raise ThreadSafetyViolationError()

    @property
    def n_parameters(self):
        """
        Number of trainable parameters of the underlying model used in the batch
        computation
        """
        return self._n_parameters

    @staticmethod
    def _validate_dimensions_not_chunked(x: da.Array):
        """
        Check if all but the first dimension are not chunked
        """
        if any([len(c) > 1 for c in x.chunks[1:]]):
            raise InvalidDimensionChunksError(x.chunks)

    @staticmethod
    def _validate_aligned_chunking(x: da.Array, y: da.Array):
        """
        Check that the chunking in the first dimensions of the two input arrays
        are aligned
        """
        if x.chunks[0] != y.chunks[0]:
            raise UnalignedChunksError(x.chunks[0], y.chunks[0])

    def influence_factors(self, x: da.Array, y: da.Array) -> da.Array:
        r"""
        Computes the expression

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradients are computed for the chunks of $(x, y)$.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise inverse
                Hessian matrix vector products for the provided batch.

        """

        self._validate_aligned_chunking(x, y)
        self._validate_dimensions_not_chunked(x)
        self._validate_dimensions_not_chunked(y)

        def func(x_numpy: NDArray, y_numpy: NDArray, model: InfluenceFunctionModel):
            factors = model.influence_factors(
                self.numpy_converter.from_numpy(x_numpy),
                self.numpy_converter.from_numpy(y_numpy),
            )
            return self.numpy_converter.to_numpy(factors)

        chunks = []
        for x_chunk, y_chunk, chunk_size in zip(
            x.to_delayed(), y.to_delayed(), x.chunks[0]
        ):
            chunk_shape = (chunk_size, self.n_parameters)
            chunk_array = da.from_delayed(
                delayed(func)(
                    x_chunk.squeeze()[()],
                    y_chunk.squeeze()[()],
                    self.influence_function_model,
                ),
                dtype=x.dtype,
                shape=chunk_shape,
            )
            chunks.append(chunk_array)

        return da.concatenate(chunks)

    def influences(
        self,
        x_test: da.Array,
        y_test: da.Array,
        x: Optional[da.Array] = None,
        y: Optional[da.Array] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> da.Array:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})),
        \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The computation is done block-wise
        for the chunks of the provided dask arrays.

        Args:
            x_test: model input to use in the gradient computations of
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise scalar
                products for the provided batch.

        """

        self._validate_aligned_chunking(x_test, y_test)
        self._validate_dimensions_not_chunked(x_test)
        self._validate_dimensions_not_chunked(y_test)

        if (x is None) != (y is None):
            if x is None:
                raise ValueError(
                    "Providing labels y without providing model input x "
                    "is not supported"
                )
            if y is None:
                raise ValueError(
                    "Providing model input x without labels y is not supported"
                )
        elif x is not None:
            self._validate_aligned_chunking(x, y)
            self._validate_dimensions_not_chunked(x)
            self._validate_dimensions_not_chunked(y)
        else:
            x, y = x_test, y_test
        assert x is not None and y is not None  # For the type checker's benefit

        def func(
            x_test_numpy: NDArray,
            y_test_numpy: NDArray,
            x_numpy: NDArray,
            y_numpy: NDArray,
            model: InfluenceFunctionModel,
        ):
            values = model.influences(
                self.numpy_converter.from_numpy(x_test_numpy),
                self.numpy_converter.from_numpy(y_test_numpy),
                self.numpy_converter.from_numpy(x_numpy),
                self.numpy_converter.from_numpy(y_numpy),
                mode,
            )
            return self.numpy_converter.to_numpy(values)

        un_chunked_x_shapes = [s[0] for s in x_test.chunks[1:]]
        x_test_chunk_sizes = x_test.chunks[0]
        x_chunk_sizes = x.chunks[0]
        blocks = []
        block_shape: Tuple[int, ...]

        for x_test_chunk, y_test_chunk, test_chunk_size in zip(
            x_test.to_delayed(), y_test.to_delayed(), x_test_chunk_sizes
        ):
            row = []
            for x_chunk, y_chunk, chunk_size in zip(
                x.to_delayed(),
                y.to_delayed(),
                x_chunk_sizes,  # type:ignore
            ):
                if mode == InfluenceMode.Up:
                    block_shape = (test_chunk_size, chunk_size)
                elif mode == InfluenceMode.Perturbation:
                    block_shape = (test_chunk_size, chunk_size, *un_chunked_x_shapes)
                else:
                    raise UnsupportedInfluenceModeException(mode)

                block_array = da.from_delayed(
                    delayed(func)(
                        x_test_chunk.squeeze()[()],
                        y_test_chunk.squeeze()[()],
                        x_chunk.squeeze()[()],
                        y_chunk.squeeze()[()],
                        self.influence_function_model,
                    ),
                    shape=block_shape,
                    dtype=x_test.dtype,
                )

                if mode == InfluenceMode.Perturbation:
                    n_dims = block_array.ndim
                    new_order = tuple(range(2, n_dims)) + (0, 1)
                    block_array = block_array.transpose(new_order)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if mode == InfluenceMode.Perturbation:
            n_dims = values_array.ndim
            new_order = (n_dims - 2, n_dims - 1) + tuple(range(n_dims - 2))
            values_array = values_array.transpose(new_order)

        return values_array

    def influences_from_factors(
        self,
        z_test_factors: da.Array,
        x: da.Array,
        y: da.Array,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> da.Array:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}},
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}},
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant
        to be per sample of the batch $(x, y)$.

        Args:
            z_test_factors: pre-computed array, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            x: optional model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
          [dask.array.Array][dask.array.Array] representing the element-wise scalar
            product of the provided batch

        """
        self._validate_aligned_chunking(x, y)
        self._validate_dimensions_not_chunked(x)
        self._validate_dimensions_not_chunked(y)
        self._validate_dimensions_not_chunked(z_test_factors)

        def func(
            z_test_numpy: NDArray,
            x_numpy: NDArray,
            y_numpy: NDArray,
            model: InfluenceFunctionModel,
        ):
            ups = model.influences_from_factors(
                self.numpy_converter.from_numpy(z_test_numpy),
                self.numpy_converter.from_numpy(x_numpy),
                self.numpy_converter.from_numpy(y_numpy),
                mode=mode,
            )
            return self.numpy_converter.to_numpy(ups)

        un_chunked_x_shape = [s[0] for s in x.chunks[1:]]
        x_chunk_sizes = x.chunks[0]
        z_test_chunk_sizes = z_test_factors.chunks[0]
        blocks = []
        block_shape: Tuple[int, ...]

        for z_test_chunk, z_test_chunk_size in zip(
            z_test_factors.to_delayed(), z_test_chunk_sizes
        ):
            row = []
            for x_chunk, y_chunk, chunk_size in zip(
                x.to_delayed(), y.to_delayed(), x_chunk_sizes
            ):
                if mode == InfluenceMode.Perturbation:
                    block_shape = (z_test_chunk_size, chunk_size, *un_chunked_x_shape)
                elif mode == InfluenceMode.Up:
                    block_shape = (z_test_chunk_size, chunk_size)
                else:
                    raise UnsupportedInfluenceModeException(mode)

                block_array = da.from_delayed(
                    delayed(func)(
                        z_test_chunk.squeeze()[()],
                        x_chunk.squeeze()[()],
                        y_chunk.squeeze()[()],
                        self.influence_function_model,
                    ),
                    shape=block_shape,
                    dtype=z_test_factors.dtype,
                )

                if mode == InfluenceMode.Perturbation:
                    n_dims = block_array.ndim
                    new_order = tuple(range(2, n_dims)) + (0, 1)
                    block_array = block_array.transpose(*new_order)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if mode == InfluenceMode.Perturbation:
            n_dims = values_array.ndim
            new_order = (n_dims - 2, n_dims - 1) + tuple(range(n_dims - 2))
            values_array = values_array.transpose(*new_order)

        return values_array

    @staticmethod
    def _get_client() -> Optional[distributed.Client]:
        try:
            return distributed.get_client()
        except ValueError:
            return None


class SequentialInfluenceCalculator:
    """
    This class serves as a simple wrapper for processing batches of data
    in a sequential manner. It is particularly useful in scenarios where parallel
    or distributed processing is not required or not feasible.
    The core functionality of this class is to apply a specified influence computation
    model, of type [InfluenceFunctionModel]
    [pydvl.influence.base_influence_function_model.InfluenceFunctionModel], to batches of data
    one at a time.

    Args:
        influence_function_model: An instance of type
                [InfluenceFunctionModel]
                [pydvl.influence.base_influence_function_model.InfluenceFunctionModel], that
                specifies the computation logic for influence on data chunks.

    Example:
        ```python
        from pydvl.influence import SequentialInfluenceCalculator
        from pydvl.influence.torch.util import (
        NestedTorchCatAggregator,
        TorchNumpyConverter,
        )
        from pydvl.influence.torch import CgInfluence

        batch_size = 10
        train_dataloader = DataLoader(..., batch_size=batch_size)
        test_dataloader = DataLoader(..., batch_size=batch_size)

        infl_model = CgInfluence(model, loss, hessian_regularization=0.01)
        infl_model = infl_model.fit(train_dataloader)

        infl_calc = SequentialInfluenceCalculator(if_model)

        # this does not trigger the computation
        lazy_influences = infl_calc.influences(test_dataloader, train_dataloader)

        # trigger computation and pull the result into main memory, result is the full
        # tensor for all combinations of the two loaders
        influences = lazy_influences.compute(aggregator=NestedTorchCatAggregator())
        # or
        # trigger computation and write results chunk-wise to disk using zarr in a
        # sequential manner
        lazy_influences.to_zarr("local_path/or/url", TorchNumpyConverter())
        ```
    """

    def __init__(
        self,
        influence_function_model: InfluenceFunctionModel,
    ):
        self.influence_function_model = influence_function_model

    def _influence_factors_gen(
        self, data_iterable: Iterable[Tuple[TensorType, TensorType]]
    ) -> Generator[TensorType, None, None]:
        for x, y in iter(data_iterable):
            yield self.influence_function_model.influence_factors(x, y)

    def influence_factors(
        self,
        data_iterable: Iterable[Tuple[TensorType, TensorType]],
    ) -> LazyChunkSequence:
        r"""
        Compute the expression

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradient are computed for the chunks $(x, y)$ of the data_iterable in
        a sequential manner.

        Args:
            data_iterable: An iterable that returns tuples of tensors.
                Each tuple consists of a pair of tensors (x, y), representing input data
                and corresponding targets.

        Returns:
            A lazy data structure representing the chunks of the resulting tensor
        """
        try:
            len_iterable = len(cast(Sized, data_iterable))
        except Exception as e:
            logger.debug(f"Failed to retrieve len of data iterable: {e}")
            len_iterable = None

        tensors_gen_factory = partial(self._influence_factors_gen, data_iterable)
        return LazyChunkSequence(tensors_gen_factory, len_generator=len_iterable)

    def _influences_gen(
        self,
        test_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        mode: InfluenceMode,
    ) -> Generator[Generator[TensorType, None, None], None, None]:
        for x_test, y_test in iter(test_data_iterable):
            yield (
                self.influence_function_model.influences(x_test, y_test, x, y, mode)
                for x, y in iter(train_data_iterable)
            )

    def influences(
        self,
        test_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> NestedLazyChunkSequence:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})),
        \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The computation is done block-wise for
        the chunks of the provided
        data iterables and aggregated into a single tensor in memory.

        Args:
            test_data_iterable: An iterable that returns tuples of tensors.
                Each tuple consists of a pair of tensors (x, y), representing input data
                and corresponding targets.
            train_data_iterable: An iterable that returns tuples of tensors.
                Each tuple consists of a pair of tensors (x, y), representing input data
                and corresponding targets.
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            A lazy data structure representing the chunks of the resulting tensor

        """
        nested_tensor_gen_factory = partial(
            self._influences_gen,
            test_data_iterable,
            train_data_iterable,
            mode,
        )

        try:
            len_iterable = len(cast(Sized, test_data_iterable))
        except Exception as e:
            logger.debug(f"Failed to retrieve len of test data iterable: {e}")
            len_iterable = None

        return NestedLazyChunkSequence(
            nested_tensor_gen_factory, len_outer_generator=len_iterable
        )

    def _influences_from_factors_gen(
        self,
        z_test_factors: Iterable[TensorType],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        mode: InfluenceMode,
    ):
        for z_test_factor in iter(z_test_factors):
            if isinstance(z_test_factor, list) or isinstance(z_test_factor, tuple):
                z_test_factor = z_test_factor[0]
            yield (
                self.influence_function_model.influences_from_factors(
                    z_test_factor, x, y, mode
                )
                for x, y in iter(train_data_iterable)
            )

    def influences_from_factors(
        self,
        z_test_factors: Iterable[TensorType],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> NestedLazyChunkSequence:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}}, \nabla_{\theta} \ell(y, f_{\theta}(x))
            \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}}, \nabla_{x} \nabla_{\theta}
            \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant to be per sample
        of the batch $(x, y)$.

        Args:
            z_test_factors: Pre-computed iterable of tensors, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            train_data_iterable: An iterable that returns tuples of tensors.
                Each tuple consists of a pair of tensors (x, y), representing input data
                and corresponding targets.
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
          A lazy data structure representing the chunks of the resulting tensor

        """
        nested_tensor_gen = partial(
            self._influences_from_factors_gen,
            z_test_factors,
            train_data_iterable,
            mode,
        )

        try:
            len_iterable = len(cast(Sized, z_test_factors))
        except Exception as e:
            logger.debug(f"Failed to retrieve len of factors iterable: {e}")
            len_iterable = None

        return NestedLazyChunkSequence(
            nested_tensor_gen, len_outer_generator=len_iterable
        )
