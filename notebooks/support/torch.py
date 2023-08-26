import logging
import os
import pickle as pkl
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from pydvl.influence.torch import as_tensor
from pydvl.utils import maybe_progress

from .types import Losses

logger = logging.getLogger(__name__)

from numpy.typing import NDArray

MODEL_PATH = Path().resolve().parent / "data" / "models"


class TorchLogisticRegression(nn.Module):
    """
    A simple binary logistic regression model.
    """

    def __init__(
        self,
        n_input: int,
    ):
        """
        :param n_input: Number of features in the input.
        """
        super().__init__()
        self.fc1 = nn.Linear(n_input, 1, bias=True, dtype=float)

    def forward(self, x):
        """
        :param x: Tensor [NxD], with N the batch length and D the number of features.
        :returns: A tensor [N] representing the probability of the positive class for each sample.
        """
        x = torch.as_tensor(x)
        return torch.sigmoid(self.fc1(x))


class TorchMLP(nn.Module):
    """
    A simple fully-connected neural network
    """

    def __init__(
        self,
        layers_size: List[int],
    ):
        """
        :param layers_size: list of integers representing the number of
            neurons in each layer.
        """
        super().__init__()
        if len(layers_size) < 2:
            raise ValueError(
                "Passed layers_size has less than 2 values. "
                "The network needs at least input and output sizes."
            )
        layers = []
        for frm, to in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(frm, to))
            layers.append(nn.Tanh())
        layers.pop()

        layers.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.
        :param x: Tensor input of shape [NxD], with N batch size and D number of
            features.
        :returns: Tensor output of shape[NxK], with K the output size of the network.
        """
        return self.layers(x)


def fit_torch_model(
    model: nn.Module,
    training_data: DataLoader,
    val_data: DataLoader,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 1,
    progress: bool = True,
) -> Losses:
    """
    Fits a pytorch model to the supplied data.
    Represents a simple machine learning loop, iterating over a number of
    epochs, sampling data with a certain batch size, calculating gradients and updating the parameters through a
    loss function.
    :param model: A pytorch model.
    :param training_data: A pytorch DataLoader with the training data.
    :param val_data: A pytorch DataLoader with the validation data.
    :param optimizer: Select either ADAM or ADAM_W.
    :param scheduler: A pytorch scheduler. If None, no scheduler is used.
    :param num_epochs: Number of epochs to repeat training.
    :param progress: True, iff progress shall be printed.
    """
    train_loss = []
    val_loss = []

    for epoch in maybe_progress(range(num_epochs), progress, desc="Model fitting"):
        batch_loss = []
        for train_batch in training_data:
            batch_x, batch_y = train_batch
            pred_y = model(batch_x)
            loss_value = loss(torch.squeeze(pred_y), torch.squeeze(batch_y))
            batch_loss.append(loss_value.item())

            logger.debug(f"Epoch: {epoch} ---> Training loss: {loss_value.item()}")
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()
        with torch.no_grad():
            batch_val_loss = []
            for val_batch in val_data:
                batch_x, batch_y = val_batch
                pred_y = model(batch_x)
                batch_val_loss.append(
                    loss(torch.squeeze(pred_y), torch.squeeze(batch_y)).item()
                )

        mean_epoch_train_loss = np.mean(batch_loss)
        mean_epoch_val_loss = np.mean(batch_val_loss)
        train_loss.append(mean_epoch_train_loss)
        val_loss.append(mean_epoch_val_loss)
        logger.info(
            f"Epoch: {epoch} ---> Training loss: {mean_epoch_train_loss}, Validation loss: {mean_epoch_val_loss}"
        )
    return Losses(train_loss, val_loss)


def new_resnet_model(output_size: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune final few layers
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, output_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


class TrainingManager:
    """A simple class to handle persistence of the model for the notebook
    `influence_imagenet.ipynb`
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        loss: torch.nn.modules.loss._Loss,
        train_data: DataLoader,
        val_data: DataLoader,
        data_dir: Path,
    ):
        self.name = name
        self.model = model
        self.loss = loss
        self.train_data = train_data
        self.val_data = val_data
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def train(
        self,
        n_epochs: int,
        lr: float = 0.001,
        use_cache: bool = True,
    ) -> Losses:
        """
        :return: Tuple of training_loss, validation_loss
        """
        if use_cache:
            try:
                losses = self.load()
                print("Cached model found, loading...")
                return losses
            except:
                print(f"No pretrained model found. Training for {n_epochs} epochs:")

        optimizer = Adam(self.model.parameters(), lr=lr)

        losses = fit_torch_model(
            model=self.model,
            training_data=self.train_data,
            val_data=self.val_data,
            loss=self.loss,
            optimizer=optimizer,
            num_epochs=n_epochs,
        )
        if use_cache:
            self.save(losses)
        self.model.eval()
        return losses

    def save(self, losses: Losses):
        """Saves the model weights and training and validation losses.

        :param training_loss: list of training losses, one per epoch
        :param validation_loss: list of validation losses, also one per epoch
        """
        torch.save(self.model.state_dict(), self.data_dir / f"{self.name}_weights.pth")
        with open(self.data_dir / f"{self.name}_train_val_loss.pkl", "wb") as file:
            pkl.dump(losses, file)

    def load(self) -> Losses:
        """Loads model weights and training and validation losses.
        :return: two arrays, one with training and one with validation losses.
        """
        self.model.load_state_dict(
            torch.load(self.data_dir / f"{self.name}_weights.pth")
        )
        self.model.eval()
        with open(self.data_dir / f"{self.name}_train_val_loss.pkl", "rb") as file:
            return pkl.load(file)


def process_imgnet_io(
    df: pd.DataFrame, labels: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = df["normalized_images"]
    y = df["labels"]
    ds_label_to_model_label = {
        ds_label: idx for idx, ds_label in enumerate(labels.values())
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_nn = torch.stack(x.tolist()).to(device)
    y_nn = torch.tensor([ds_label_to_model_label[yi] for yi in y], device=device)
    return x_nn, y_nn
