from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial, wraps
from typing import Generic, Optional, Type

from ..utils.progress import log_duration
from .array import LazyChunkSequence, SumAggregator
from .types import BatchType, BlockMapperType, DataLoaderType, InfluenceMode, TensorType


class UnsupportedInfluenceModeException(ValueError):
    def __init__(self, mode: str):
        super().__init__(
            f"Provided {mode=} is not supported. Choose one of InfluenceMode.Up "
            f"and InfluenceMode.Perturbation"
        )


class NotFittedException(ValueError):
    def __init__(self, object_type: Type):
        super().__init__(
            f"Objects of type {object_type} must be fitted before calling "
            f"methods. "
            f"Call method fit with appropriate input."
        )


class NotImplementedLayerRepresentationException(ValueError):
    def __init__(self, module_id: str):
        message = f"Only Linear layers are supported, but found module {module_id} requiring grad."
        super().__init__(message)


class InfluenceFunctionModel(Generic[TensorType, DataLoaderType], ABC):
    """
    Generic abstract base class for computing influence related quantities.
    For a specific influence algorithm and tensor framework,
    inherit from this base class
    """

    @property
    @abstractmethod
    def n_parameters(self):
        """Number of trainable parameters of the underlying model"""

    @property
    @abstractmethod
    def is_thread_safe(self) -> bool:
        """Whether the influence computation is thread safe"""

    @property
    @abstractmethod
    def is_fitted(self):
        """Override this, to expose the fitting status of the instance."""

    @abstractmethod
    def fit(self, data: DataLoaderType) -> InfluenceFunctionModel:
        """
        Override this method to fit the influence function model to training data,
        e.g. pre-compute hessian matrix or matrix decompositions

        Args:
            data:

        Returns:
            The fitted instance
        """

    @staticmethod
    def fit_required(method):
        """Decorator to enforce the fitted check"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self.is_fitted:
                raise NotFittedException(type(self))
            return method(self, *args, **kwargs)

        return wrapper

    def influence_factors(self, x: TensorType, y: TensorType) -> TensorType:
        if not self.is_fitted:
            raise NotFittedException(type(self))
        return self._influence_factors(x, y)

    @abstractmethod
    def _influence_factors(self, x: TensorType, y: TensorType) -> TensorType:
        r"""
        Override this method to implement the approximation of

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradient is meant to be per sample of the batch $(x, y)$.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            Tensor representing the element-wise inverse Hessian matrix vector products

        """

    def influences(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: Optional[TensorType] = None,
        y: Optional[TensorType] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> TensorType:
        if not self.is_fitted:
            raise NotFittedException(type(self))

        if x is None and y is not None:
            raise ValueError(
                "Providing labels y, without providing model input x "
                "is not supported"
            )

        if x is not None and y is None:
            raise ValueError(
                "Providing model input x, without providing labels y "
                "is not supported"
            )

        return self._influences(x_test, y_test, x, y, mode)

    @abstractmethod
    def _influences(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: Optional[TensorType] = None,
        y: Optional[TensorType] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> TensorType:
        r"""
        Override this method to implement the approximation of

        \[ \langle H^{-1}\nabla_{theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case.

        Args:
            x_test: model input to use in the gradient computations
                of $H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{test}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """

    @abstractmethod
    def influences_from_factors(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> TensorType:
        r"""
        Override this method to implement the computation of

        \[ \langle z_{\text{test_factors}},
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}},
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant to be per sample
        of the batch $(x, y)$.

        Args:
            z_test_factors: pre-computed array, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                f_{\theta}(x_{\text{test}}))$
            x: model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """


class ComposableInfluence(
    InfluenceFunctionModel,
    Generic[TensorType, BatchType, DataLoaderType, BlockMapperType],
    ABC,
):

    block_mapper: BlockMapperType

    @property
    def is_fitted(self):
        try:
            return self.block_mapper is not None
        except AttributeError:
            return False

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoaderType) -> InfluenceFunctionModel:
        self.block_mapper = self._create_block_mapper(data)
        return self

    @abstractmethod
    def _create_block_mapper(self, data: DataLoaderType) -> BlockMapperType:
        pass

    @InfluenceFunctionModel.fit_required
    def influences_by_block(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: Optional[TensorType] = None,
        y: Optional[TensorType] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> OrderedDict[str, TensorType]:
        left_batch = self._create_batch(x_test, y_test)

        if x is None:
            if y is not None:
                raise ValueError(
                    "Providing labels y, without providing model input x "
                    "is not supported"
                )
            right_batch = left_batch
        else:
            if y is None:
                raise ValueError(
                    "Providing model input x, without providing labels y "
                    "is not supported"
                )
            right_batch = self._create_batch(x, y)

        return self.block_mapper.block_interactions(left_batch, right_batch, mode)

    @InfluenceFunctionModel.fit_required
    def influence_factors_by_block(
        self, x: TensorType, y: TensorType
    ) -> OrderedDict[str, TensorType]:
        return self.block_mapper.block_transformed_gradients(self._create_batch(x, y))

    @InfluenceFunctionModel.fit_required
    def influences_from_factors_by_block(
        self,
        z_test_factors: OrderedDict[str, TensorType],
        x: TensorType,
        y: TensorType,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> OrderedDict[str, TensorType]:
        return self.block_mapper.block_interactions_from_transformed_gradients(
            z_test_factors, self._create_batch(x, y), mode
        )

    def _influence_factors(self, x: TensorType, y: TensorType) -> TensorType:
        tensor_gen_factory = partial(
            self.block_mapper.generate_transformed_gradients, self._create_batch(x, y)
        )
        aggregator = SumAggregator()
        result: TensorType = aggregator(LazyChunkSequence(tensor_gen_factory))
        return result

    def _influences(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: Optional[TensorType] = None,
        y: Optional[TensorType] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> TensorType:
        left_batch = self._create_batch(x_test, y_test)

        if x is None:
            right_batch = left_batch
        elif y is None:
            raise ValueError(
                "Providing model input x, without providing labels y "
                "is not supported"
            )
        else:
            right_batch = self._create_batch(x, y)

        tensor_gen_factory = partial(
            self.block_mapper.generate_gradient_interactions,
            left_batch,
            right_batch,
            mode,
        )
        aggregator = SumAggregator()
        result: TensorType = aggregator(LazyChunkSequence(tensor_gen_factory))
        return result

    @InfluenceFunctionModel.fit_required
    def influences_from_factors(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> TensorType:
        tensor_gen_factory = partial(
            self.block_mapper.generate_interactions_from_transformed_gradients,
            z_test_factors,
            self._create_batch(x, y),
            mode,
        )

        aggregator = SumAggregator()
        result: TensorType = aggregator(LazyChunkSequence(tensor_gen_factory))
        return result

    @staticmethod
    @abstractmethod
    def _create_batch(x: TensorType, y: TensorType) -> BatchType:
        pass
