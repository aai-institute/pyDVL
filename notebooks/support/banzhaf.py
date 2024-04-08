from typing import Optional

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from numpy.typing import NDArray
from pydvl.utils.types import SupervisedModel


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
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required in order to load the Digits Dataset"
        ) from e

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
        lr: float,
        epochs: int,
        batch_size: int,
        device: str,
    ):
        self.lr = lr
        self.batch_size = batch_size
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
        self.epochs = epochs
        self.model.to(device)

    def fit(self, x: NDArray, y: NDArray) -> None:
        torch_dataset = TensorDataset(
            torch.tensor(
                np.reshape(x, (x.shape[0], 1, 8, 8)), dtype=torch.float, device=device
            ),
            torch.tensor(y, device=device),
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
                np.reshape(x, (x.shape[0], 1, 8, 8)), dtype=torch.float, device=device
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
