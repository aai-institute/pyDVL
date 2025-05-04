r"""
Tests for tensor support in ClasswiseShapleyValuation
"""

from __future__ import annotations

import numpy as np
import pytest

from pydvl.utils.array import try_torch_import
from pydvl.valuation import FiniteNoIndexIteration
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods import ClasswiseShapleyValuation
from pydvl.valuation.samplers import (
    ClasswiseSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
)
from pydvl.valuation.scorers import ClasswiseSupervisedScorer
from pydvl.valuation.stopping import MaxUpdates
from pydvl.valuation.utility import ClasswiseModelUtility
from tests.valuation.methods.conftest import TorchLinearClassifier

torch = try_torch_import()
if torch is None:
    pytest.skip("PyTorch not available", allow_module_level=True)


def test_classwise_shapley_tensor_support(
    tensor_classwise_utility, tensor_train_dataset
):
    """Test that ClasswiseShapleyValuation works with tensor inputs."""
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration
    )
    sampler = ClasswiseSampler(
        in_class=in_class_sampler,
        out_of_class=out_of_class_sampler,
        batch_size=1,
    )

    # Create the valuation with tensor inputs
    valuation = ClasswiseShapleyValuation(
        tensor_classwise_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )

    # Fit the valuation
    valuation.fit(tensor_train_dataset)

    # Check result
    assert valuation.result is not None
    assert isinstance(
        valuation.result.values, np.ndarray
    )  # Result values are always numpy arrays
    assert (
        len(valuation.result.values) == 4
    )  # Should have a value for each training point

    # Ensure all values are finite
    assert np.all(np.isfinite(valuation.result.values))


def test_compare_tensor_and_numpy_results():
    """Test that tensor and numpy inputs produce equivalent results."""
    # Create numpy datasets
    x_train_np = np.arange(1, 5).reshape(-1, 1).astype(np.float32)
    y_train_np = np.array([0, 0, 1, 1], dtype=np.int64)
    np_train_dataset = Dataset(x_train_np, y_train_np)

    x_test_np = x_train_np.copy()
    y_test_np = np.array([0, 0, 0, 1], dtype=np.int64)
    np_test_dataset = Dataset(x_test_np, y_test_np)

    # Create numpy-based utility
    from tests.valuation.methods.test_classwise_shapley import (
        ClosedFormLinearClassifier,
    )

    np_utility = ClasswiseModelUtility(
        ClosedFormLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", np_test_dataset),
        catch_errors=False,
    )

    # Create tensor datasets
    x_train_tensor = torch.tensor(x_train_np)
    y_train_tensor = torch.tensor(y_train_np)
    tensor_train_dataset = Dataset(x_train_tensor, y_train_tensor)

    x_test_tensor = torch.tensor(x_test_np)
    y_test_tensor = torch.tensor(y_test_np)
    tensor_test_dataset = Dataset(x_test_tensor, y_test_tensor)

    # Create tensor-based utility
    tensor_utility = ClasswiseModelUtility(
        TorchLinearClassifier(),
        ClasswiseSupervisedScorer("accuracy", tensor_test_dataset),
        catch_errors=False,
    )

    # Create samplers
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration
    )
    sampler = ClasswiseSampler(
        in_class=in_class_sampler,
        out_of_class=out_of_class_sampler,
        batch_size=1,
    )

    # Create valuations
    np_valuation = ClasswiseShapleyValuation(
        np_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )

    tensor_valuation = ClasswiseShapleyValuation(
        tensor_utility,
        sampler=sampler,
        is_done=MaxUpdates(100),
        progress=False,
        normalize_values=True,
    )

    # Fit valuations
    np_valuation.fit(np_train_dataset)
    tensor_valuation.fit(tensor_train_dataset)

    # Compare results - they should be close but not identical due to floating point differences
    np.testing.assert_allclose(
        np_valuation.result.values, tensor_valuation.result.values, rtol=1e-5, atol=1e-5
    )
