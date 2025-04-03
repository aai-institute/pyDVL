from __future__ import annotations

import numpy as np
import pytest

from pydvl.valuation import (
    ConstantSampleSize,
    Dataset,
    DeltaShapleyNCSGDConfig,
    DeltaShapleyNCSGDSampleSize,
    HarmonicSampleSize,
    MinUpdates,
    PowerLawSampleSize,
    RandomIndexIteration,
    RandomSizeIteration,
    StratifiedSampler,
    ValuationResult,
)
from pydvl.valuation.methods import DeltaShapleyValuation
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.base import UtilityBase
from tests.valuation import check_values, recursive_make


def _make_constant_result(n_indices: int, value: float, name: str) -> ValuationResult:
    return ValuationResult(
        values=np.full(n_indices, value),
        variances=np.zeros(n_indices),
        counts=np.ones(n_indices),
        data_names=np.arange(n_indices),
        algorithm=name,
    )


class SubsetSizeConstantUtility(UtilityBase):
    """A utility that returns a constant value per subset size.

    The utility of a coalition (i.e. subset) $S$ of size $k=|S|$ is defined as

    $$u(S) = k (k-1).$$

    With this definition it is possible to compute analytically the value of any point
    i:

    * `v_shap(i) = n-1`
    * `v_delta(i) = (u^2-l^2+u+l)/(u-l+1)`
    """

    def __call__(self, sample: Sample | None) -> float:
        k = len(sample.subset) if sample else 0
        return k * (k - 1)

    @staticmethod
    def exact_delta_shapley(
        n_indices: int, lower_bound: int, upper_bound: int
    ) -> ValuationResult:
        value = (upper_bound**2 - lower_bound**2 + upper_bound + lower_bound) / (
            upper_bound - lower_bound + 1
        )
        return _make_constant_result(n_indices, value, "exact-delta-shapley")

    @staticmethod
    def exact_shapley(n_indices: int) -> ValuationResult:
        value = n_indices - 1
        return _make_constant_result(n_indices, value, "exact-shapley")

    @staticmethod
    def exact_shapley_clipped(
        lower_bound: int, upper_bound: int, n_indices: int
    ) -> ValuationResult:
        """Shapley value when certain subset sizes are discarded, e.g. with a
        [StratifiedSampler][pydvl.valuation.samplers.stratified.StratifiedSampler].
        """
        value = (
            upper_bound**2 - lower_bound**2 + upper_bound + lower_bound
        ) / n_indices
        return _make_constant_result(n_indices, value, "exact-shapley-clipped")


@pytest.mark.parametrize(
    "sample_sizes_cls, sample_sizes_kwargs",
    [
        (ConstantSampleSize, {"n_samples": lambda n: n}),
        (PowerLawSampleSize, {"n_samples": lambda n: n, "exponent": -0.5}),
        (HarmonicSampleSize, {"n_samples": lambda n: n}),
        (
            DeltaShapleyNCSGDSampleSize,
            {
                "config": (
                    DeltaShapleyNCSGDConfig,
                    dict(
                        # these are all just dummies
                        lipschitz_grad=1,
                        lipschitz_loss=1,
                        lr_factor=1,
                        n_sgd_iter=1,
                        max_loss=1,
                        n_val=lambda n: n,
                        n_train=lambda n: n,
                        eps=0.5,
                        delta=0.5,
                        version="theorem7",
                    ),
                )
            },
        ),
    ],
)
@pytest.mark.parametrize("n_indices, lower_bound, upper_bound", [(5, 1, 4)])
def test_delta_shapley(
    sample_sizes_cls, sample_sizes_kwargs, n_indices, lower_bound, upper_bound, seed
):
    # dummy datasets, since the utility does not use any data, only set sizes
    train, test = Dataset.from_arrays(
        X=np.zeros((2 * n_indices, 1)), y=np.zeros(2 * n_indices), train_size=n_indices
    )

    utility = SubsetSizeConstantUtility()
    sample_sizes_kwargs |= {"lower_bound": lower_bound, "upper_bound": upper_bound}
    sample_size_strategy = recursive_make(
        sample_sizes_cls,
        sample_sizes_kwargs,
        n_samples=n_indices,
        # dummy, for NC-SGD
        n_val=len(test),
        n_train=len(train),
    )
    m = sample_size_strategy.n_samples_per_index(n_indices)
    n_updates = 2 ** (n_indices + 1) if m is None else m * n_indices

    valuation = DeltaShapleyValuation(
        utility=utility,
        sampler=StratifiedSampler(
            sample_sizes=sample_size_strategy,
            sample_sizes_iteration=RandomSizeIteration,
            index_iteration=RandomIndexIteration,
            batch_size=n_indices,
            seed=seed,
        ),
        is_done=MinUpdates(n_updates),
        progress=False,
        skip_converged=False,
    )

    valuation.fit(train)
    result = valuation.values()
    exact_result = SubsetSizeConstantUtility.exact_delta_shapley(
        n_indices, lower_bound, upper_bound
    )
    check_values(result, exact_result, rtol=0.1, atol=0.1)
