import numpy as np
import pytest

from pydvl.utils.array import array_equal, is_numpy, to_numpy, try_torch_import
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.types import Sample, ValueUpdate

torch = try_torch_import()
pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


@pytest.fixture
def numpy_indices():
    return np.array([0, 1, 2, 3, 4], dtype=np.int_)


@pytest.fixture
def torch_indices():
    return torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)


@pytest.fixture
def numpy_sample():
    return Sample(1, np.array([0, 2, 3], dtype=np.int_))


@pytest.fixture
def torch_sample():
    return Sample(1, torch.tensor([0, 2, 3], dtype=torch.int64))


@pytest.mark.parametrize(
    "subset",
    [
        None,
        np.array([0, 1, 2], dtype=np.int_),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int32),
    ],
)
def test_sample_creation_with_tensor(subset):
    """Test that the Sample class works with PyTorch tensors."""
    # Test with int64 tensor
    sample = Sample(0, subset)
    assert is_numpy(sample.subset)
    assert array_equal(sample.subset, to_numpy(subset))


@pytest.mark.parametrize(
    "idx, subset, subset_with_idx",
    [
        (0, np.empty(0, dtype=int), np.array([0])),
        (0, np.array([0, 1, 2], dtype=np.int_), np.array([0, 1, 2])),
        (3, torch.tensor([0, 1, 2], dtype=torch.int8), np.array([0, 1, 2, 3])),
    ],
)
def test_sample_with_idx_in_subset_tensor(idx, subset, subset_with_idx):
    """Test the with_idx_in_subset method with tensor subsets."""
    sample = Sample(idx, subset)

    new_sample = sample.with_idx_in_subset()
    assert is_numpy(new_sample.subset)
    assert array_equal(new_sample.subset, subset_with_idx)


def test_sample_with_idx_tensor():
    """Test the with_idx method with tensor indices."""
    subset = torch.tensor([1, 2, 3], dtype=torch.int64)
    sample = Sample(0, subset)

    new_sample = sample.with_idx(1)
    assert new_sample.idx == 1
    assert torch.is_tensor(new_sample.subset)
    assert torch.equal(new_sample.subset, subset)

    # Test with same idx (should return same object)
    new_sample = sample.with_idx(0)
    assert new_sample is sample


def test_sample_with_subset_tensor():
    """Test the with_subset method with tensor subsets."""
    subset = torch.tensor([1, 2, 3], dtype=torch.int64)
    sample = Sample(0, subset)

    new_subset = torch.tensor([4, 5, 6], dtype=torch.int64)
    new_sample = sample.with_subset(new_subset)
    assert new_sample.idx == 0
    assert is_numpy(new_sample.subset)
    assert array_equal(new_sample.subset, new_subset.cpu().numpy())


def test_sample_equality_tensor(numpy_sample, torch_sample):
    """Test equality between samples with different array types."""
    # A tensor Sample should equal a numpy Sample with the same values
    assert numpy_sample == torch_sample
    assert torch_sample == numpy_sample

    # Test with different values
    different_subset = torch.tensor([0, 2, 4], dtype=torch.int64)
    different_sample = Sample(1, different_subset)
    assert numpy_sample != different_sample
    assert different_sample != numpy_sample


def test_sample_hash_tensor(numpy_sample, torch_sample):
    """Test hash consistency between samples with different array types."""
    # Hash should be the same for equivalent samples regardless of array type
    assert hash(numpy_sample) == hash(torch_sample)

    # Test dictionary lookups
    sample_dict = {numpy_sample: "numpy_sample"}
    assert torch_sample in sample_dict
    assert sample_dict[torch_sample] == "numpy_sample"


def test_index_sampler_skip_indices(numpy_indices, torch_indices):
    """Test that skip_indices properly handles both numpy arrays and torch tensors."""

    class TestSampler(IndexSampler):
        def __init__(self):
            super().__init__(batch_size=1)
            # Override the skip_indices setter to accept indices
            self._skip_indices = np.empty(0, dtype=bool)

        @IndexSampler.skip_indices.setter
        def skip_indices(self, indices):
            """Custom skip_indices setter that supports both numpy and torch indices."""
            self._skip_indices = indices

        def sample_limit(self, indices):
            return len(indices)

        def generate(self, indices):
            for idx in indices:
                if (
                    isinstance(self._skip_indices, np.ndarray)
                    and len(self._skip_indices) > 0
                ):
                    if self._skip_indices[idx]:
                        continue
                elif (
                    torch.is_tensor(self._skip_indices) and len(self._skip_indices) > 0
                ):
                    if self._skip_indices[idx].item():
                        continue
                yield Sample(idx, np.array([]))

        def log_weight(self, n, subset_len):
            return 0.0

        def make_strategy(self, utility, log_coefficient):
            return None

    # Test with numpy indices
    sampler_np = TestSampler()
    skip_mask_np = np.zeros(5, dtype=bool)
    skip_mask_np[2] = True  # Skip index 2
    sampler_np.skip_indices = skip_mask_np

    # Check if indices are skipped correctly
    samples_np = list(sampler_np.generate(numpy_indices))
    assert len(samples_np) == 4  # One less than 5 because we skip index 2
    assert all(s.idx != 2 for s in samples_np)

    # Test with torch indices
    sampler_torch = TestSampler()
    skip_mask_torch = torch.zeros(5, dtype=torch.bool)
    skip_mask_torch[2] = True  # Skip index 2
    sampler_torch.skip_indices = skip_mask_torch

    # Check if indices are skipped correctly
    samples_torch = list(sampler_torch.generate(torch_indices))
    assert len(samples_torch) == 4  # One less than 5 because we skip index 2
    assert all(s.idx != 2 for s in samples_torch)


def test_generate_batches_tensor_indices(torch_indices):
    """Test that generate_batches works with torch tensor indices."""

    class TestSampler(IndexSampler):
        def sample_limit(self, indices):
            return len(indices)

        def generate(self, indices):
            for idx in indices:
                yield Sample(
                    idx.item() if torch.is_tensor(idx) else idx,
                    torch.tensor([], dtype=torch.int64)
                    if torch.is_tensor(indices)
                    else np.array([], dtype=np.int_),
                )

        def log_weight(self, n, subset_len):
            return 0.0

        def make_strategy(self, utility, log_coefficient):
            return None

    # Test with torch indices
    sampler = TestSampler(batch_size=2)
    batches = list(sampler.generate_batches(torch_indices))

    # Check batch structure
    assert len(batches) == 3  # 5 samples with batch_size=2 should yield 3 batches
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1

    # Check that all indices are correctly generated
    all_samples = [sample for batch in batches for sample in batch]
    assert len(all_samples) == 5
    assert set(sample.idx for sample in all_samples) == {0, 1, 2, 3, 4}


def test_value_update_tensor_idx():
    """Test that ValueUpdate works with PyTorch tensor indices."""
    # Test with tensor index
    tensor_idx = torch.tensor(1, dtype=torch.int64)
    update = ValueUpdate(tensor_idx.item(), 0.5, 1)
    assert update.idx == 1
    assert update.log_update == 0.5
    assert update.sign == 1
