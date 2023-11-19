from itertools import groupby
from typing import Any, Callable, Optional, Tuple

import distributed
import numpy as np
from dask import array as da
from numpy.typing import NDArray

from ..twice_differentiable import Influence, InfluenceType, InverseHvpResult

__all__ = ["DaskInfluence"]


class DaskInfluence(Influence[da.Array]):
    """
    Compute influences over dask.Array collections. Depends on a batch computation model
    of type [Influence][pydvl.influence.twice_differentiable.Influence]. In addition, provide transformations
    from and to numpy, corresponding to the tensor types of the batch computation model.
    Args:
        influence_model: instance of type [Influence][pydvl.influence.twice_differentiable.Influence], defines the
            batch-wise computation model
        to_numpy: transformation for turning the tensor type output of a batch computation into a numpy array
        from_numpy: transformation for turning numpy arrays into the correct tensor type to apply the batch
            computation model
    """

    def __init__(
        self,
        influence_model: Influence,
        to_numpy: Callable[[Any], np.ndarray],
        from_numpy: Callable[[np.ndarray], Any],
    ):
        self.from_numpy = from_numpy
        self.to_numpy = to_numpy
        self._num_parameters = influence_model.num_parameters
        self.influence_model = influence_model.prepare_for_distributed()
        client = self._get_client()
        if client is not None:
            self.influence_model = client.scatter(influence_model, broadcast=True)

    @property
    def num_parameters(self):
        """Number of trainable parameters of the underlying model used in the batch computation"""
        return self._num_parameters

    @staticmethod
    def _validate_un_chunked(x: da.Array):
        if any([len(c) > 1 for c in x.chunks[1:]]):
            raise ValueError("Array must be un-chunked in ever dimension but the first")

    @staticmethod
    def _validate_aligned_chunking(x: da.Array, y: da.Array):
        if x.chunks[0] != y.chunks[0]:
            raise ValueError(
                "x and y must have the same chunking in the first dimension"
            )

    @staticmethod
    def _get_chunk_indices(
        chunk_sizes: Tuple[int, ...], aggregate_same_chunk_size: bool = False
    ) -> Tuple[Tuple[int, int], ...]:
        indices = []
        start = 0

        if aggregate_same_chunk_size:
            for value, group in groupby(chunk_sizes):
                length = sum(group)
                indices.append((start, start + length))
                start += length
        else:
            for value in chunk_sizes:
                indices.append((start, start + value))
                start += value

        return tuple(indices)

    def factors(self, x: da.Array, y: da.Array) -> InverseHvpResult[da.Array]:
        """
        Compute the expression
        $$
        H^{-1}\nabla_{theta} \ell(y, f_{\theta}(x))
        $$
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
        return InverseHvpResult(self._factors_without_info(x, y), {})

    def _factors_without_info(self, x: da.Array, y: da.Array):
        def func(x_numpy: NDArray, y_numpy: NDArray, model: Influence):
            factors, _ = model.factors(
                self.from_numpy(x_numpy), self.from_numpy(y_numpy)
            )
            return self.to_numpy(factors)

        def block_func(x_block: da.Array, y_block: NDArray):
            chunk_size = x.chunks[0][0]
            return da.map_blocks(
                func,
                x_block,
                y_block,
                self.influence_model,
                dtype=x_block.dtype,
                chunks=(chunk_size, self.num_parameters),
            )

        return da.concatenate(
            [
                block_func(x[start:stop], y[start:stop])
                for (start, stop) in self._get_chunk_indices(
                    x.chunks[0], aggregate_same_chunk_size=True
                )
            ],
            axis=0,
        )

    def up_weighting(
        self, z_test_factors: da.Array, x: da.Array, y: da.Array
    ) -> da.Array:
        """
        Computation of
        $$
        \langle z_test_factors, \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        where the gradients are computed chunk-wise for the chunks of $(x, y)$.
        Args:
            z_test_factors: pre-computed array, approximating $H^{-1}\nabla_{\theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise scalar product of the provided batch

        """
        self._validate_aligned_chunking(x, y)
        self._validate_un_chunked(x)
        self._validate_un_chunked(y)
        self._validate_un_chunked(z_test_factors)

        def func(
            z_test_numpy: NDArray, x_numpy: NDArray, y_numpy: NDArray, model: Influence
        ):
            ups = model.up_weighting(
                self.from_numpy(z_test_numpy),
                self.from_numpy(x_numpy),
                self.from_numpy(y_numpy),
            )
            return self.to_numpy(ups)

        return da.blockwise(
            func,
            "ij",
            z_test_factors,
            "ik",
            x,
            "jn",
            y,
            "jm",
            model=self.influence_model,
            concatenate=True,
            dtype=x.dtype,
        )

    def perturbation(
        self, z_test_factors: da.Array, x: da.Array, y: da.Array
    ) -> da.Array:
        """
        Computation of
        $$
        \langle z_test_factors, \nabla_x \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        where the gradients are computed chunk-wise for the chunks of $z_test_factors$  and $(x, y)$.
        Args:
            z_test_factors: pre-computed array, approximating $H^{-1}\nabla_{\theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            [dask.array.Array][dask.array.Array] representing the element-wise scalar product for the provided batch

        """

        self._validate_aligned_chunking(x, y)
        self._validate_un_chunked(x)
        self._validate_un_chunked(y)
        self._validate_un_chunked(z_test_factors)

        def func(
            z_test_numpy: NDArray, x_numpy: NDArray, y_numpy: NDArray, model: Influence
        ):
            ups = model.perturbation(
                self.from_numpy(z_test_numpy),
                self.from_numpy(x_numpy),
                self.from_numpy(y_numpy),
            )
            return self.to_numpy(ups)

        return da.blockwise(
            func,
            "ijb",
            z_test_factors,
            "ik",
            x,
            "jb",
            y,
            "jm",
            model=self.influence_model,
            concatenate=True,
            align_arrays=True,
            dtype=x.dtype,
        )

    def values(
        self,
        x_test: da.Array,
        y_test: da.Array,
        x: Optional[da.Array] = None,
        y: Optional[da.Array] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> InverseHvpResult:
        """
        Compute approximation of
        $$
        \langle H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        for the case of up-weighting influence, resp.
        $$
        \langle H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test})), \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        for the perturbation type influence case. The computation is done block-wise for the chunks of the provided dask
        arrays.

        Args:
            x_test: model input to use in the gradient computations of $H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations $\nabla_{theta}\ell(y, f_{\theta}(x))$, resp. $\nabla_{x}\nabla_{theta}\ell(y, f_{\theta}(x))$, if None,
                use $x=x_{test}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
            Container object of type [InverseHvpResult][pydvl.influence.twice_differentiable.InverseHvpResult] with a
            [dask.array.Array][dask.array.Array] representing the element-wise scalar products for the provided batch.
            Retrieval of batch inversion information is not yet implemented.

        """

        self._validate_aligned_chunking(x_test, y_test)
        self._validate_un_chunked(x_test)
        self._validate_un_chunked(y_test)

        if (x is None) != (y is None):
            raise ValueError()  # ToDO error message
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
            model: Influence,
        ):
            values, _ = model.values(
                self.from_numpy(x_test_numpy),
                self.from_numpy(y_test_numpy),
                self.from_numpy(x_numpy),
                self.from_numpy(y_numpy),
                influence_type,
            )
            return self.to_numpy(values)

        resulting_shape = "ij" if influence_type is InfluenceType.Up else "ijk"
        result = da.blockwise(
            func,
            resulting_shape,
            x_test,
            "ik",
            y_test,
            "im",
            x,
            "jk",
            y,
            "jm",
            model=self.influence_model,
            concatenate=True,
            dtype=x.dtype,
            align_arrays=True,
        )
        return InverseHvpResult(result, {})

    @staticmethod
    def _get_client() -> Optional[distributed.Client]:
        try:
            return distributed.get_client()
        except ValueError:
            return None
