import numpy as np
import pytest
import torch

from pydvl.valuation.types import Sample
from pydvl.valuation.utility.deepset import DeepSet, DeepSetUtilityModel, SetDatasetRaw


class DummyData:
    def __init__(self, x: np.ndarray):
        self.x = x


class DummyDataset:
    def __init__(self, x: np.ndarray):
        self._data = DummyData(x)
        self.n_features = x.shape[1]

    def data(self):
        return self._data


@pytest.fixture(scope="session")
def device(request):
    import torch

    use_cuda = request.config.getoption("--with-cuda")
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.mark.parametrize(
    "use_embedding, input_shape, extra_args",
    [
        # Non-embedding: input tensor is (batch, set_size, input_dim)
        (False, (4, 5, 3), {}),
        # Embedding: input tensor is (batch, set_size) of integer ids.
        (True, (4, 5), {"num_embeddings": 10, "input_dim": 3}),
    ],
)
def test_deepset_forward_shape(use_embedding, input_shape, extra_args):
    if use_embedding:
        model = DeepSet(
            phi_hidden_dim=8,
            phi_output_dim=6,
            rho_hidden_dim=4,
            use_embedding=True,
            **extra_args,
        )
        x = torch.randint(0, extra_args["num_embeddings"], input_shape)
    else:
        model = DeepSet(
            input_dim=input_shape[-1],
            phi_hidden_dim=8,
            phi_output_dim=6,
            rho_hidden_dim=4,
            use_embedding=False,
        )
        x = torch.randn(input_shape)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    # Expected output shape: (batch_size, 1)
    assert output.shape == (input_shape[0], 1)


def test_deepset_invalid_embedding_params():
    # When use_embedding=True without num_embeddings or input_dim, a ValueError should be raised.
    with pytest.raises(ValueError):
        DeepSet(
            input_dim=None,
            phi_hidden_dim=8,
            phi_output_dim=6,
            rho_hidden_dim=4,
            use_embedding=True,
        )


def test_deepset_permutation_invariance(seed):
    # Test that the model is invariant to the order of set elements.
    batch_size = 2
    set_size = 7
    input_dim = 4
    model = DeepSet(
        input_dim=input_dim,
        phi_hidden_dim=4,
        phi_output_dim=3,
        rho_hidden_dim=2,
        use_embedding=False,
    )
    torch.manual_seed(seed)
    x = torch.randn((batch_size, set_size, input_dim))
    output1 = model(x)
    perm = torch.randperm(set_size)
    x_permuted = x[:, perm, :]
    output2 = model(x_permuted)
    assert torch.allclose(output1, output2, atol=1e-6)


@pytest.fixture(scope="module")
def dummy_dataset(n_features=3, n_samples=5):
    # 5 samples, each row is a feature vector.
    x = np.array(
        [[i + j for j in range(n_features)] for i in range(n_samples)], dtype=np.float32
    )
    return DummyDataset(x)


def test_setdatasetraw_getitem(dummy_dataset):
    sample1 = Sample(idx=None, subset=np.array([0, 1]))
    sample2 = Sample(idx=None, subset=np.array([2, 3, 4]))

    samples_dict = {sample1: 1.0, sample2: 2.0}
    dataset_raw = SetDatasetRaw(samples_dict, dummy_dataset)

    assert len(dataset_raw) == 2

    # Test __getitem__ for sample1.
    set_tensor, target = dataset_raw[0]
    assert set_tensor.shape == (3, dummy_dataset.n_features)

    x = dummy_dataset.data().x
    np.testing.assert_allclose(set_tensor[0].numpy(), x[0])
    np.testing.assert_allclose(set_tensor[1].numpy(), x[1])
    np.testing.assert_allclose(
        set_tensor[2].numpy(), np.zeros(dummy_dataset.n_features)
    )
    assert target.shape == (1,)
    assert target.item() == 1.0


def test_setdatasetraw_device(dummy_dataset, device):
    sample1 = Sample(idx=None, subset=np.array([0, 1]))
    sample2 = Sample(idx=None, subset=np.array([2, 3, 4]))

    samples_dict = {sample1: 1.0, sample2: 2.0}
    dataset_raw = SetDatasetRaw(samples_dict, dummy_dataset, device=device)
    assert dataset_raw[0][0].device.type == torch.device(device).type


def test_deepset_utility_model_predict():
    x = np.array([[i, i + 1, i + 2] for i in range(10)], dtype=np.float32)
    dummy_dataset = DummyDataset(x)

    sample1 = Sample(idx=None, subset=np.array([0, 1]))
    sample2 = Sample(idx=None, subset=np.array([2, 3]))
    sample3 = Sample(idx=None, subset=np.array([4, 5, 6]))
    sample4 = Sample(idx=None, subset=np.array([7, 8, 9]))

    samples_dict = {sample1: 1.0, sample2: 2.0, sample3: 3.0, sample4: 4.0}

    model = DeepSetUtilityModel(
        dummy_dataset, phi_hidden_dim=8, phi_output_dim=6, rho_hidden_dim=4
    )
    model.fit(samples_dict)

    test_samples = [sample1, sample3]
    predictions = model.predict(test_samples)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(test_samples), 1)
    assert predictions.dtype == np.float32


def test_deepset_set_device(device):
    model = DeepSet(
        input_dim=10,
        phi_hidden_dim=8,
        phi_output_dim=6,
        rho_hidden_dim=4,
        use_embedding=False,
    ).to(device=device)

    model.forward(torch.zeros((1, 1, 10), device=device))
