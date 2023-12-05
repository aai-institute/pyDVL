from math import prod
from typing import Any, Callable, Optional, Tuple

import distributed
import numpy as np
from dask import array as da
from dask import delayed
from numpy.typing import NDArray

from pydvl.influence.base_influence_model import (
    InfluenceFunctionModel,
    InfluenceType,
    UnSupportedInfluenceTypeException,
)

__all__ = ["DaskInfluenceCalculator"]


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
    of type [InfluenceFunctionModel][pydvl.influence.base_influence_mode.InfluenceFunctionModel].
    In addition, provide transformations from and to numpy,
    corresponding to the tensor types of the batch computation model.
    Args:
        influence_model: instance of type
            [InfluenceFunctionModel][pydvl.influence.base_influence_mode.InfluenceFunctionModel], defines the
            batch-wise computation model
        to_numpy: transformation for turning the tensor type output of a batch computation into a numpy array
        from_numpy: transformation for turning numpy arrays into the correct tensor type to apply the batch
            computation model
    """

    def __init__(
        self,
        influence_model: InfluenceFunctionModel,
        to_numpy: Callable[[Any], np.ndarray],
        from_numpy: Callable[[np.ndarray], Any],
    ):
        self.from_numpy = from_numpy
        self.to_numpy = to_numpy
        self._num_parameters = influence_model.num_parameters
        self.influence_model = influence_model
        client = self._get_client()
        if client is not None:
            self.influence_model = client.scatter(influence_model, broadcast=True)
        else:
            self.influence_model = delayed(influence_model)

    @property
    def num_parameters(self):
        """Number of trainable parameters of the underlying model used in the batch computation"""
        return self._num_parameters

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
            Container object of type [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult] with a
            tensor representing the element-wise inverse Hessian matrix vector products for the provided batch.
            Retrieval of batch inversion information is not yet implemented.

        """

        self._validate_aligned_chunking(x, y)
        self._validate_un_chunked(x)
        self._validate_un_chunked(y)

        def func(x_numpy: NDArray, y_numpy: NDArray, model: InfluenceFunctionModel):
            factors = model.influence_factors(
                self.from_numpy(x_numpy), self.from_numpy(y_numpy)
            )
            return self.to_numpy(factors)

        chunks = []
        for x_chunk, y_chunk, chunk_size in zip(
            x.to_delayed(), y.to_delayed(), x.chunks[0]
        ):
            chunk_shape = (chunk_size, self.num_parameters)
            chunk_array = da.from_delayed(
                delayed(func)(
                    x_chunk.squeeze().tolist(),
                    y_chunk.squeeze().tolist(),
                    self.influence_model,
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
            x_test: model input to use in the gradient computations of $H^{-1}\nabla_{\theta} \ell(y_{test},
                f_{\theta}(x_{test}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$, if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

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
                self.from_numpy(x_test_numpy),
                self.from_numpy(y_test_numpy),
                self.from_numpy(x_numpy),
                self.from_numpy(y_numpy),
                influence_type,
            )
            return self.to_numpy(values)

        un_chunked_x_length = prod([s[0] for s in x_test.chunks[1:]])
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
                    block_shape = (test_chunk_size, chunk_size, un_chunked_x_length)
                else:
                    raise UnSupportedInfluenceTypeException(influence_type)

                block_array = da.from_delayed(
                    delayed(func)(
                        x_test_chunk.squeeze().tolist(),
                        y_test_chunk.squeeze().tolist(),
                        x_chunk.squeeze().tolist(),
                        y_chunk.squeeze().tolist(),
                        self.influence_model,
                    ),
                    shape=block_shape,
                    dtype=x_test.dtype,
                )

                if influence_type == InfluenceType.Perturbation:
                    block_array = block_array.transpose(2, 0, 1)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if influence_type == InfluenceType.Perturbation:
            values_array = values_array.transpose(1, 2, 0)

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
        \[
        \langle z_{\text{test_factors}}, \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        \]
        for the case of up-weighting influence, resp.
        \[
        \langle z_{\text{test_factors}}, \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        \]
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
                self.from_numpy(z_test_numpy),
                self.from_numpy(x_numpy),
                self.from_numpy(y_numpy),
                influence_type=influence_type,
            )
            return self.to_numpy(ups)

        un_chunked_x_length = prod([s[0] for s in x.chunks[1:]])
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
                    block_shape = (z_test_chunk_size, chunk_size, un_chunked_x_length)
                elif influence_type == InfluenceType.Up:
                    block_shape = (z_test_chunk_size, chunk_size)
                else:
                    raise UnSupportedInfluenceTypeException(influence_type)

                block_array = da.from_delayed(
                    delayed(func)(
                        z_test_chunk.squeeze().tolist(),
                        x_chunk.squeeze().tolist(),
                        y_chunk.squeeze().tolist(),
                        self.influence_model,
                    ),
                    shape=block_shape,
                    dtype=z_test_factors.dtype,
                )

                if influence_type == InfluenceType.Perturbation:
                    block_array = block_array.transpose(2, 0, 1)

                row.append(block_array)
            blocks.append(row)

        values_array = da.block(blocks)

        if influence_type == InfluenceType.Perturbation:
            values_array = values_array.transpose(1, 2, 0)

        return values_array

    @staticmethod
    def _get_client() -> Optional[distributed.Client]:
        try:
            return distributed.get_client()
        except ValueError:
            return None
