from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from torch.cuda.amp import GradScaler
from tqdm import trange

from pydvl.utils import timed
from pydvl.utils.monitor import end_memory_monitoring, start_memory_monitoring
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers import SupervisedScorer
from pydvl.valuation.scorers.torchscorer import TorchModelScorer
from pydvl.valuation.types import TorchSupervisedModel
from pydvl.valuation.utility.modelutility import ModelUtility
from pydvl.valuation.utility.torchutility import TorchUtility

try:
    import torch
    from torch import Tensor, nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise RuntimeError("PyTorch is required to run the Banzhaf MSR notebook") from e

logger = logging.getLogger("notebooks.support.banzhaf")


@overload
def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    use_tensors: Literal[False] = False,
    device: str | None = "cpu",
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset[NDArray], Dataset[NDArray]]: ...


@overload
def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    use_tensors: Literal[True] = True,
    device: str | None = "cpu",
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset[Tensor], Dataset[Tensor]]: ...


def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    use_tensors: bool = True,
    device: str | None = "cpu",
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset, Dataset]:
    """Loads the sklearn handwritten digits dataset.

    More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset.

    Args:
        train_size: Fraction of points used for training.
        random_state: Fix random seed. If `None`, the default rng is taken.
        use_tensors: If `True`, the data us stored as PyTorch tensors.
        device: Device to load the data to. If `None`, the data is left on the CPU.
        shared_mem: If `True`, the tensors are moved to shared memory.
    Returns
        A tuple of tra ining and test Datasets
    """

    bunch = load_digits(as_frame=True)
    if use_tensors:
        dtype = torch.float16 if half_precision else torch.float32
        x = torch.tensor(bunch.data.values / 16.0, dtype=dtype, device=device)
        y = torch.tensor(bunch.target.values, dtype=torch.long, device=device)
        x, y = TorchClassifierModel.reshape_inputs(x, y)
        mb = (x.nbytes + y.nbytes) / 2**20
        if shared_mem:
            x.share_memory_()
            y.share_memory_()
            logger.debug(
                f"Loaded {len(x)} data points to {x.device} ({mb:.2f}MB, {shared_mem=})"
            )
    else:
        dtype = np.float16 if half_precision else np.float32
        x = np.array(bunch.data.values / 16.0, dtype=dtype)
        y = np.array(bunch.target.values, dtype=np.int32)
        mb = (x.nbytes + y.nbytes) / 2**20
        logger.debug(f"Loaded {len(x)} data points ({mb:.2f}MB, mmap={shared_mem})")

    train, test = Dataset.from_arrays(
        x,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify_by_target=True,
        mmap=shared_mem and not use_tensors,
        target_names=["label"],
    )
    # TODO: Is this helping?
    del x, y
    import gc

    gc.collect()
    return train, test


class SimpleCNN(nn.Module):
    """A simple CNN model for the MNIST dataset.
    Input is (batch_size, 1, 8, 8) and output is (batch_size, 10).
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding="same"
            ),
            nn.Conv2d(
                in_channels=8, out_channels=4, kernel_size=(3, 3), padding="same"
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TorchClassifierModel(TorchSupervisedModel):
    """This class wraps a torch classification model to comply with the
    [SupervisedModel][pydvl.utils.types.SupervisedModel] interface expected by pyDVL,
    and takes care of the training and evaluation of the model.

    !!! note
        This is just a PoC. Loss function and optimizer choice are hard-coded, there
        is no validation, no early-stopping, no learning rate scheduling, etc.

    Args:
        model: A PyTorch nn.Module to be trained.
        lr: Learning rate for the Adam optimizer.
        epochs: Number of epochs to train the model.
        batch_size: Size of the batches used for training.
        device: Device to use for training. Can be 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        lr: float,
        epochs: int,
        batch_size: int,
        device: torch.device | str,
    ):
        self.model_factory = model_factory
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = None  # TODO
        self.model = None

    def fit(self, x: Tensor, y: Tensor):
        """Runs the training loop for the model.

        Note that the utility object will have cloned the model before calling us, so we
        need to move the model to the device here.
        """
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError(
                f"x and y must be PyTorch tensors, were {type(x)}, {type(y)}"
            )
        self.model = self.make_model()
        device = next(self.model.parameters()).device
        assert x.device == y.device == device, (
            f"Data and model not on the same device: {x.device}, {y.device} and {device}. "
            f"Should be {self.device}."
        )

        torch_dataset = TensorDataset(x, y)
        torch_dataloader = DataLoader(
            torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            for features, labels in torch_dataloader:
                optimizer.zero_grad()
                pred = self.model(features)
                loss = self.loss(pred, labels)
                loss.backward()
                optimizer.step()

    def fit_amp(self, x: Tensor, y: Tensor) -> None:
        """Runs the training loop for the model.

        Note that the utility object will have cloned the model before calling us, so we
        need to move the model to the device here.
        """
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError(
                f"x and y must be PyTorch tensors, were {type(x)}, {type(y)}"
            )
        _p = next(self.model.parameters())
        device = _p.device
        dtype = _p.dtype
        self.model = self.make_model()
        logger.debug(f"Using autocast for {dtype}")
        torch_dataset = TensorDataset(x, y)
        torch_dataloader = DataLoader(torch_dataset, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scaler = GradScaler()
        for epoch in range(self.epochs):
            for features, labels in torch_dataloader:
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=dtype):
                    pred = self.model(features)
                    loss = self.loss(pred, labels)
                # Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
                # first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped
                scaler.step(optimizer)
                scaler.update()

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a PyTorch tensor, was {type(x)}")
        pred = self.model(x)
        return torch.argmax(pred, dim=1)

    @torch.inference_mode()
    def predict_amp(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a PyTorch tensor, was {type(x)}")
        with torch.autocast(device_type=x.device.type, dtype=x.dtype):
            pred = self.model(x)
            return torch.argmax(pred, dim=1)

    @torch.inference_mode()
    def score(self, x: Tensor, y: Tensor) -> float:
        pred = self.predict(x)
        acc = torch.sum(pred == y).float() / y.shape[0]
        return acc.cpu().item()

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {
            "model_factory": self.model_factory,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    @staticmethod
    def reshape_inputs(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return x.reshape((x.shape[0], 1, 8, 8)), y

    def make_model(self) -> torch.nn.Module:
        """Creates a new model instance.

        This is used to create a new model for each evaluation of the utility.
        """
        return self.model_factory().to(device=self.device)


def move_optimizer_to_device(optimizer: optim.Optimizer, device: str | torch.device):
    """Moves the optimizer state to the given device.

    Helpful if the model is moved to a different device after the optimizer was created.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, Tensor) and v.device != device:
                logger.info(f"Moving optimizer state {k} to {device}")
                state[k] = v.to(device)


from sacred import Experiment
from sacred.observers import TinyDbObserver

from pydvl.valuation import MinUpdates, PermutationSampler, RelativeTruncation
from pydvl.valuation.methods import BanzhafValuation

ex = Experiment("bzf_torch_utility", save_git_info=False)
ex.observers.append(TinyDbObserver("bzf_torch_utility"))


@ex.config
def config():
    n_jobs = 6  # noqa: F841
    min_updates = 100  # noqa: F841
    truncation_rtol = 0.05  # noqa: F841
    truncation_burn_in_fraction = 0.3  # noqa: F841

    lr = 0.01  # noqa: F841
    n_epochs = 24  # noqa: F841
    batch_size = 256  # noqa: F841
    model_device = "cuda"  # noqa: F841
    data_device = "cuda"  # noqa: F841
    half_precision = False  # noqa: F841
    load_as_tensors = True  # noqa: F841
    shared_mem = False  # noqa: F841
    custom_utility = False  # noqa: F841
    custom_scorer = False  # noqa: F841
    clone_before_fit = True  # noqa: F841

    memory_monitor = False  # noqa: F841
    catch_errors = False  # noqa: F841
    show_warnings = False  # noqa: F841


@ex.automain
def run(_config):
    if _config["memory_monitor"]:
        start_memory_monitoring()

    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s", force=True
    )

    train, test = load_digits_dataset(
        0.7,
        random_state=_config["seed"],
        use_tensors=_config["load_as_tensors"],
        device=_config["data_device"],
        shared_mem=_config["shared_mem"],
        half_precision=_config["half_precision"],
    )

    model = TorchClassifierModel(
        model_factory=SimpleCNN,
        lr=_config["lr"],
        epochs=_config["n_epochs"],
        batch_size=_config["batch_size"],
        device=_config["model_device"],
    )
    if _config["custom_scorer"]:
        scorer = TorchModelScorer(model, test, default=0.0, range=(0.0, 1.0))
    else:
        scorer = SupervisedScorer(model, test, default=0.0, range=(0.0, 1.0))

    if _config["custom_utility"]:
        utility = TorchUtility(
            model,
            scorer,
            catch_errors=_config["catch_errors"],
            show_warnings=_config["show_warnings"],
            clone_before_fit=_config["clone_before_fit"],
        )
    else:
        utility = ModelUtility(
            model,
            scorer,
            catch_errors=_config["catch_errors"],
            show_warnings=_config["show_warnings"],
            clone_before_fit=_config["clone_before_fit"],
        )

    truncation = RelativeTruncation(
        rtol=_config["truncation_rtol"],
        burn_in_fraction=_config["truncation_burn_in_fraction"],
    )
    sampler = PermutationSampler(truncation=truncation)
    stopping = MinUpdates(_config["min_updates"])

    valuation = BanzhafValuation(
        utility,
        sampler=sampler,
        is_done=stopping,
        progress=True,
        show_warnings=_config["show_warnings"],
    )

    # timed_fit = timed(model.fit)
    # timed_score = timed(accumulate=True)(model.score)
    # timed_fit(*train.data()[:600])
    # print(f"Training completed in {timed_fit.execution_time:.3f} s")
    # print(f"Training accuracy: {timed_score(*train.data()):.3f}")
    # print(f"Test accuracy: {timed_score(*test.data()):.3f}")
    # print(f"Evaluation completed in {timed_score.execution_time:.3f} s")

    timed_fit = timed(accumulate=True)(model.fit)
    timed_score = timed(accumulate=True)(model.score)
    training_accuracies = []
    test_accuracies = []
    n_points = len(train)
    for n in trange(n_points // 3, n_points):
        timed_fit(*train.data()[:n])
        training_accuracies.append(timed_score(*train.data()[:n]))
        test_accuracies.append(timed_score(*test.data()))
    n_runs = len(training_accuracies)
    print(f"Training accuracy: {np.mean(training_accuracies):.3f}")
    print(f"Test accuracy: {np.mean(test_accuracies):.3f}")
    print(f"Average training time: {timed_fit.execution_time / n_runs:.3f} s")

    import matplotlib.pyplot as plt

    plt.plot(training_accuracies, label="Training accuracy")
    plt.plot(test_accuracies, label="Test accuracy")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # timed_fit = timed(valuation.fit)
    # with parallel_config(backend="loky", n_jobs=_config["n_jobs"]):
    #     timed_fit(train)

    # joblib.dump(valuation.values(), "bzf_values.pkl")
    # ex.add_artifact("bzf_values.pkl")

    end_memory_monitoring()
    return timed_fit.execution_time
