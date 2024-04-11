import numpy as np

from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Optional

from pydvl.utils.types import SupervisedModel

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    raise RuntimeError("PyTorch is required to run the Banzhaf MSR notebook") from e


def load_digits_dataset(
    test_size: float, val_size: float = 0.0, random_state: Optional[int] = None
):
    """Loads the sklearn handwritten digits dataset. More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset.

    :param test_size: fraction of points used for test dataset
    :param val_size: fraction of points used for training dataset
    :param random_state: fix random seed. If None, no random seed is set.
    :return: A tuple of three elements with the first three being input and
        target values in the form of matrices of shape (N,8,8) the first
        and (N,) the second.
    """

    digits_bunch = load_digits(as_frame=True)
    x, x_test, y, y_test = train_test_split(
        digits_bunch.data.values / 16.0,
        digits_bunch.target.values,
        train_size=1 - test_size,
        random_state=random_state,
    )
    if val_size > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=(1 - val_size) / (1 - test_size), random_state=random_state
        )
    else:
        x_train, y_train = x, y
        x_val, y_val = None, None

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))


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
                out_channels=8, in_channels=1, kernel_size=(3, 3), padding="same"
            ),
            nn.Conv2d(
                out_channels=4, in_channels=8, kernel_size=(3, 3), padding="same"
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=10),
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
