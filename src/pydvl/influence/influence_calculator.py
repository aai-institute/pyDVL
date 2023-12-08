"""
This module provides functionality for calculating influences for large amount of data. The computation
is based on a chunk computation model in the form of an instance of
[InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel], which is mapped
over collection of chunks.

Classes:
    DaskInfluenceCalculator: Core class for computing influences over dask.Array collections.
    SequentialInfluenceCalculator: A class for sequential processing of influence calculations.

"""


from functools import partial
from typing import Generator, Iterable, Optional, Tuple

import distributed
from dask import array as da
from dask import delayed
from numpy.typing import NDArray

from .array import NumpyConverter, OneDimChunkedLazyArray, TwoDimChunkedLazyArray
from .base_influence_model import (
    InfluenceFunctionModel,
    InfluenceType,
    TensorType,
    UnSupportedInfluenceTypeException,
)

__all__ = ["DaskInfluenceCalculator", "SequentialInfluenceCalculator"]


class DimensionChunksException(ValueError):
    def __init__(self, chunks: Tuple[Tuple[int, ...], ...]):
        msg = (
            f"Array must be un-chunked in every dimension but the first, got {chunks=}"
        )
        super().__init__(msg)


class UnalignedChunksException(ValueError):
    def __init__(self, chunk_sizes_x: Tuple[int, ...], chunk_sizes_y: Tuple[int, ...]):
        msg = (
            f"Arrays x and y must have the same chunking in the first dimension, got {chunk_sizes_x=} "
            f"and {chunk_sizes_y=}"
        )
        super().__init__(msg)


class DaskInfluenceCalculator:
    """
    Compute influences over dask.Array collections. Depends on a batch computation model
    of type [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel].
    In addition, provide transformations from and to numpy,
    corresponding to the tensor types of the batch computation model.
    Args:
        influence_function_model: instance of type
            [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel], defines the
            batch-wise computation model
        numpy_converter: instance of type [NumpyConverter][pydvl.influence.influence_calculator.NumpyConverter], used to
            convert between numpy arrays and TensorType objects needed to use the underlying model

    !!! Warning
        Make sure, you set `threads_per_worker=1`, when using the distributed scheduler for computing
        if your implementation of [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel]
        is not thread-safe. If you do not use the distributed scheduler,
        choose the `processes` single machine scheduler
        ```python
        client = Client(threads_per_worker=1)
        # or
        da_influences.compute(scheduler="processes")
        ```
        For details on dask schedulers see the [official documentation](https://docs.dask.org/en/stable/scheduling.html).


    """

    def __init__(
        self,
        influence_function_model: InfluenceFunctionModel,
        numpy_converter: NumpyConverter,
    ):
        self._n_parameters = influence_function_model.n_parameters
        self.influence_function_model = influence_function_model
        self.numpy_converter = numpy_converter
        client = self._get_client()
        if client is not None:
            self.influence_function_model = client.scatter(
                influence_function_model, broadcast=True
            )
        else:
            self.influence_function_model = delayed(influence_function_model)

    @property
    def n_parameters(self):
        """Number of trainable parameters of the underlying model used in the batch computation"""
        return self._n_parameters

    @staticmethod
    def _validate_un_chunked(x: da.Array):
        if any([len(c) > 1 for c in x.chunks[1:]]):
            raise DimensionChunksException(x.chunks)

    @staticmethod
    def _validate_aligned_chunking(x: da.Array, y: da.Array):
        if x.chunks[0] != y.chunks[0]:
            raise UnalignedChunksException(x.chunks[0], y.chunks[0])

    def influence_factors(self, x: da.Array, y: da.Array) -> da.Array:
        r"""
        Compute the expression

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradient are computed for the chunks of $(x, y)$.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise inverse Hessian matrix vector
                products for the provided batch.

        """

        self._validate_aligned_chunking(x, y)
        self._validate_un_chunked(x)
        self._validate_un_chunked(y)

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
                    x_chunk.squeeze().tolist(),
                    y_chunk.squeeze().tolist(),
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
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> da.Array:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
        \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The computation is done block-wise for the chunks of the provided dask
        arrays.

        Args:
            x_test: model input to use in the gradient computations of $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$, if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.base_influence_model.InfluenceType]

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise scalar products for the provided batch.

        """

        self._validate_aligned_chunking(x_test, y_test)
        self._validate_un_chunked(x_test)
        self._validate_un_chunked(y_test)

        if (x is None) != (y is None):
            if x is None:
                raise ValueError(
                    "Providing labels y without providing model input x is not supported"
                )
            if y is None:
                raise ValueError(
                    "Providing model input x without labels y is not supported"
                )
        elif x is not None:
            self._validate_aligned_chunking(x, y)
            self._validate_un_chunked(x)
            self._validate_un_chunked(y)
        else:
            x, y = x_test, y_test

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
                influence_type,
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
                x.to_delayed(), y.to_delayed(), x_chunk_sizes  # type:ignore
            ):
                if influence_type == InfluenceType.Up:
                    block_shape = (test_chunk_size, chunk_size)
                elif influence_type == InfluenceType.Perturbation:
                    block_shape = (test_chunk_size, chunk_size, *un_chunked_x_shapes)
                else:
                    raise UnSupportedInfluenceTypeException(influence_type)

                block_array = da.from_delayed(
                    delayed(func)(
                        x_test_chunk.squeeze().tolist(),
                        y_test_chunk.squeeze().tolist(),
                        x_chunk.squeeze().tolist(),
                        y_chunk.squeeze().tolist(),
                        self.influence_function_model,
                    ),
                    shape=block_shape,
                    dtype=x_test.dtype,
                )

                if influence_type == InfluenceType.Perturbation:
                    n_dims = block_array.ndim
                    new_order = tuple(range(2, n_dims)) + (0, 1)
                    block_array = block_array.transpose(new_order)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if influence_type == InfluenceType.Perturbation:
            n_dims = values_array.ndim
            new_order = (n_dims - 2, n_dims - 1) + tuple(range(n_dims - 2))
            values_array = values_array.transpose(new_order)

        return values_array

    def influences_from_factors(
        self,
        z_test_factors: da.Array,
        x: da.Array,
        y: da.Array,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> da.Array:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}}, \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}}, \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant to be per sample of the batch $(x, y)$.

        Args:
            z_test_factors: pre-computed array, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}}))$
            x: optional model input to use in the gradient computations $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$, if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
          [dask.array.Array][dask.array.Array] representing the element-wise scalar product of the provided batch

        """
        self._validate_aligned_chunking(x, y)
        self._validate_un_chunked(x)
        self._validate_un_chunked(y)
        self._validate_un_chunked(z_test_factors)

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
                influence_type=influence_type,
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
                if influence_type == InfluenceType.Perturbation:
                    block_shape = (z_test_chunk_size, chunk_size, *un_chunked_x_shape)
                elif influence_type == InfluenceType.Up:
                    block_shape = (z_test_chunk_size, chunk_size)
                else:
                    raise UnSupportedInfluenceTypeException(influence_type)

                block_array = da.from_delayed(
                    delayed(func)(
                        z_test_chunk.squeeze().tolist(),
                        x_chunk.squeeze().tolist(),
                        y_chunk.squeeze().tolist(),
                        self.influence_function_model,
                    ),
                    shape=block_shape,
                    dtype=z_test_factors.dtype,
                )

                if influence_type == InfluenceType.Perturbation:
                    n_dims = block_array.ndim
                    new_order = tuple(range(2, n_dims)) + (0, 1)
                    block_array = block_array.transpose(*new_order)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if influence_type == InfluenceType.Perturbation:
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
    Simple wrapper class to process batches of data sequentially. Depends on a batch computation model
    of type [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel].

    Args:
    influence_function_model: instance of type
        [InfluenceFunctionModel][pydvl.influence.base_influence_model.InfluenceFunctionModel], defines the
        batch-wise computation model
    block_aggregator: optional instance of type [BlockAggregator][pydvl.influence.influence_calculator.BlockAggregator],
        used to collect and aggregate the tensors from the sequential process. If None, tensors are collected into
        list structures
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
    ) -> OneDimChunkedLazyArray:
        r"""
        Compute the expression

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradient are computed for the chunks $(x, y)$ of the data_iterable in a sequential manner and
        aggregated into a single tensor.

        Args:
            data_iterable:

        Returns:
            Tensor representing the element-wise inverse Hessian matrix vector
                products for the provided batch.

        """
        tensors_gen_factory = partial(self._influence_factors_gen, data_iterable)
        return OneDimChunkedLazyArray(tensors_gen_factory)

    def _influences_gen(
        self,
        test_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        influence_type: InfluenceType,
    ) -> Generator[Generator[TensorType, None, None], None, None]:

        for x_test, y_test in iter(test_data_iterable):
            yield (
                self.influence_function_model.influences(
                    x_test, y_test, x, y, influence_type
                )
                for x, y in iter(train_data_iterable)
            )

    def influences(
        self,
        test_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> TwoDimChunkedLazyArray:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
        \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The computation is done block-wise for the chunks of the provided
        data iterables and aggregated into a single tensor in memory.

        Args:

            test_data_iterable:
            train_data_iterable:
            influence_type: enum value of [InfluenceType][pydvl.influence.base_influence_model.InfluenceType]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch.

        """
        nested_tensor_gen_factory = partial(
            self._influences_gen,
            test_data_iterable,
            train_data_iterable,
            influence_type,
        )

        return TwoDimChunkedLazyArray(nested_tensor_gen_factory)

    def _influences_from_factors_gen(
        self,
        z_test_factors: Iterable[TensorType],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        influence_type: InfluenceType,
    ):

        for z_test_factor in iter(z_test_factors):
            if isinstance(z_test_factor, list) or isinstance(z_test_factor, tuple):
                z_test_factor = z_test_factor[0]
            yield (
                self.influence_function_model.influences_from_factors(
                    z_test_factor, x, y, influence_type
                )
                for x, y in iter(train_data_iterable)
            )

    def influences_from_factors(
        self,
        z_test_factors: Iterable[TensorType],
        train_data_iterable: Iterable[Tuple[TensorType, TensorType]],
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> TwoDimChunkedLazyArray:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}}, \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}}, \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant to be per sample of the batch $(x, y)$.

        Args:
            z_test_factors: pre-computed iterable of tensors, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}}))$
            train_data_iterable:
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
          Tensor representing the element-wise scalar product of the provided batch

        """
        nested_tensor_gen = partial(
            self._influences_from_factors_gen,
            z_test_factors,
            train_data_iterable,
            influence_type,
        )
        return TwoDimChunkedLazyArray(nested_tensor_gen)
