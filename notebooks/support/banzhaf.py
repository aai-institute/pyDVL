from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from pydvl.utils.types import SupervisedModel
from pydvl.valuation.dataset import Dataset

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise RuntimeError("PyTorch is required to run the Banzhaf MSR notebook") from e


def load_digits_dataset(train_size: float, random_state: Optional[int] = None):
    """Loads the sklearn handwritten digits dataset.

    More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset.

    Args:
        train_size: Fraction of points used for training.
        random_state: Fix random seed. If None, no random seed is set.

    Returns
        A tuple of three elements with the first three being input and target values in the form of matrices of shape (N, 8, 8) the first and (N,) the second.
    """

    digits_bunch = load_digits(as_frame=True)
    train, test = Dataset.from_arrays(
        digits_bunch.data.values / 16.0,
        digits_bunch.target.values,
        train_size=train_size,
        random_state=random_state,
    )
    return train, test


class TorchCNNModel(SupervisedModel):
    def __init__(
        self,
        lr: float = 0.001,
        epochs: int = 40,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(
                out_channels=8, in_channels=1, kernel_size=(5, 5), padding="same"
            ),
            nn.Conv2d(
                out_channels=4, in_channels=8, kernel_size=(3, 3), padding="same"
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1),
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(device)

    def fit(self, x: NDArray, y: NDArray) -> None:
        torch_dataset = TensorDataset(
            torch.tensor(
                np.reshape(x, (x.shape[0], 1, 8, 8)),
                dtype=torch.float,
                device=self.device,
            ),
            torch.tensor(y, device=self.device),
        )
        torch_dataloader = DataLoader(torch_dataset, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            for features, labels in torch_dataloader:
                pred = self.model(features)
                loss = self.loss(pred, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, x: NDArray) -> NDArray:
        pred = self.model(
            torch.tensor(
                np.reshape(x, (x.shape[0], 1, 8, 8)),
                dtype=torch.float,
                device=self.device,
            )
        )
        pred = torch.argmax(pred, dim=1)
        return pred.cpu().numpy()

    def score(self, x: NDArray, y: NDArray) -> float:
        pred = self.predict(x)
        acc = accuracy_score(pred, y)
        return acc

    def get_params(self, deep: bool = False):
        return {"lr": self.lr, "epochs": self.epochs}
