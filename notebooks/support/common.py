from __future__ import annotations

import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skorch
from matplotlib import gridspec
from matplotlib.axes import Axes
from numpy.typing import NDArray
from PIL.JpegImagePlugin import JpegImageFile
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pydvl.utils.array import try_torch_import
from pydvl.valuation.dataset import Dataset

from .influence import Losses


def plot_gaussian_blobs(
    train_ds: Tuple[NDArray[np.float64], NDArray[np.int_]],
    test_ds: Tuple[NDArray[np.float64], NDArray[np.int_]],
    x_min: Optional[NDArray[np.float64]] = None,
    x_max: Optional[NDArray[np.float64]] = None,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    vline: Optional[float] = None,
    line: Optional[NDArray[np.float64]] = None,
    suptitle: Optional[str] = None,
    s: Optional[float] = None,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plots training and test data in two separate plots, with the optimal
    decision boundary as passed to the line argument.

    Args:
        train_ds: A 2-element tuple with training data and labels thereof.
            Features have shape `(N, 2)` and the target_variable has shape `(n,)`.
        test_ds: A 2-element tuple with test data and labels. Same format as
            train_ds.
        x_min: Set to define the minimum boundaries of the plot.
        x_max: Set to define the maximum boundaries of the plot.
        line: Optional, line of shape (M,2), where each row is a point of the
            2-d line.
        s: The thickness of the points to plot.
        figsize: for `plt.figure()`
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(20, 2)
    ax = [fig.add_subplot(spec[:-1, i]) for i in range(2)]
    ax.append(fig.add_subplot(spec[-1, :]))

    datasets = {"train": train_ds, "test": test_ds}

    discrete_keys = [
        key for key, dataset in datasets.items() if dataset[1].dtype == int
    ]
    if 0 < len(discrete_keys) < len(datasets):
        "You can only plot either discrete or only continuous plots."

    cmap = plt.get_cmap("Set1")
    all_y = np.concatenate(tuple([v[1] for _, v in datasets.items()]), axis=0)
    unique_y = np.sort(np.unique(all_y))
    num_classes = len(unique_y)
    handles = [mpatches.Patch(color=cmap(i), label=y) for i, y in enumerate(unique_y)]

    for i, dataset_name in enumerate(datasets.keys()):
        x, y = datasets[dataset_name]
        if x.shape[1] != 2:
            raise AttributeError("The maximum number of allowed features is 2.")

        ax[i].set_title(dataset_name)
        if x_min is not None:
            ax[i].set_xlim(x_min[0], x_max[0])  # type: ignore
        if x_max is not None:
            ax[i].set_ylim(x_min[1], x_max[1])  # type: ignore

        if line is not None:
            ax[i].plot(line[:, 0], line[:, 1], color="black")

        ax[i].scatter(x[:, 0], x[:, 1], c=cmap(y), s=s, edgecolors="black")

        if xlabel is not None:
            ax[i].set_xlabel(xlabel)
        if ylabel is not None:
            ax[i].set_ylabel(ylabel)

        if vline is not None:
            ax[i].axvline(vline, color="black", linestyle="--")

    ax[-1].legend(handles=handles, loc="center", ncol=num_classes, title=legend_title)
    ax[-1].axis("off")

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.show()


def plot_influences(
    x: NDArray[np.float64],
    influences: NDArray[np.float64],
    corrupted_indices: Optional[List[int]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    line: Optional[NDArray[np.float64]] = None,
    suptitle: Optional[str] = None,
    colorbar_limits: Optional[Tuple] = None,
) -> plt.Axes:
    """Plots the influence values of the training data with a color map.

    Args:
        x: Input to the model, of shape (N,2) with N being the total number
            of points.
        influences: an array  of shape (N,) with influence values for each
            data point.
        line: Optional, line of shape [Mx2], where each row is a point of the
            2-dimensional line. (??)
    """
    if ax is None:
        _, ax = plt.subplots()
    sc = ax.scatter(x[:, 0], x[:, 1], c=influences)
    if line is not None:
        ax.plot(line[:, 0], line[:, 1], color="black")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.colorbar(sc, label=legend_title)
    if colorbar_limits is not None:
        sc.clim(*colorbar_limits)
    if corrupted_indices is not None:
        ax.scatter(
            x[:, 0][corrupted_indices],
            x[:, 1][corrupted_indices],
            facecolors="none",
            edgecolors="r",
            s=80,
        )
    return ax


def plot_iris(
    data: Dataset,
    indices: list[int] = None,
    highlight_indices: Sequence[int] | None = None,
    suptitle: str = None,
    legend_title: str = None,
    legend_labels: Sequence[str] = None,
    colors: Iterable = None,
    colorbar_limits: tuple | None = None,
    figsize: tuple[int, int] = (20, 8),
):
    """Scatter plots for the iris dataset.

    Args:
        data: a Dataset with a valid train / test split
        indices: subset of `data.indices`.
        highlight_indices: circle these indices in red
        suptitle: centered title for the figure
        legend_title: A title for the legend
        legend_labels: Labels for the legend entries
        colors: use with indices to set the color (e.g. to values).
        colorbar_limits: Range of values to display in the colorbar. A
            colorbar will only be displayed if there are more than 10 colors.
        figsize: Size of figure for matplotlib
    """
    x_train, y_train = data[indices].data()

    sepal_length_indices = data.feature("sepal length (cm)")
    sepal_width_indices = data.feature("sepal width (cm)")
    petal_length_indices = data.feature("petal length (cm)")
    petal_width_indices = data.feature("petal width (cm)")

    if colors is None:
        colors = y_train
    marker_size = 2 * plt.rcParams["lines.markersize"] ** 2

    is_value_plot = len(np.unique(colors)) > 10

    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle, fontweight="bold")

    if is_value_plot:  # Two columns for the plots and a narrow one for the colorbar
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.3)
        ax0 = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[2])
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

    def plot_scatter(
        txt: str,
        indices_a,
        indices_b,
        highlight_indices,
        add_legend: bool,
        ax: Axes,
        ticks_right: bool,
    ):
        if ticks_right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        xmin, xmax = x_train[indices_a].min(), x_train[indices_a].max()
        ymin, ymax = x_train[indices_b].min(), x_train[indices_b].max()
        xmargin = 0.1 * (xmax - xmin)
        ymargin = 0.1 * (ymax - ymin)
        sc = ax.scatter(
            x_train[indices_a],
            x_train[indices_b],
            c=colors,
            s=marker_size,
            marker="o",
            alpha=0.8,
        )
        ax.set_xlim(xmin - xmargin, xmax + xmargin)
        ax.set_ylim(ymin - ymargin, ymax + ymargin)
        ax.set_xlabel(f"{txt} length")
        ax.set_ylabel(f"{txt} width")
        if add_legend:
            ax.legend(
                handles=sc.legend_elements()[0],
                labels=legend_labels,
                title=legend_title,
            )

        if highlight_indices is not None:
            ax.scatter(
                x_train[indices_a][highlight_indices],
                x_train[indices_b][highlight_indices],
                facecolors="none",
                edgecolors="r",
                s=marker_size * 1.1,
            )
        return sc

    plot_scatter(
        "Sepal",
        sepal_length_indices,
        sepal_width_indices,
        highlight_indices,
        not is_value_plot,
        ax0,
        ticks_right=False,
    )
    sc1 = plot_scatter(
        "Petal",
        petal_length_indices,
        petal_width_indices,
        highlight_indices,
        not is_value_plot,
        ax1,
        ticks_right=True,
    )

    if is_value_plot:
        cb = fig.colorbar(sc1, cax=cax)
        cb.set_label(legend_title)
        cb.ax.yaxis.label.set_rotation(0)
        cb.ax.yaxis.set_label_coords(0.4, -0.03)  # yikes
        if colorbar_limits is not None:
            sc1.set_clim(*colorbar_limits)


def plot_sample_images(dataset: pd.DataFrame, n_images_per_class: int = 3):
    """Plots several images for each class of a pre-processed imagenet dataset
    (or a subset of it).

    Args:
        dataset: imagenet dataset
        n_images_per_class: number of images per class to plot
    """
    labels = dataset["labels"].unique()
    fig, axes = plt.subplots(nrows=n_images_per_class, ncols=len(labels))
    fig.suptitle("Examples of training images")
    for class_idx, class_label in enumerate(labels):
        for img_idx, (_, img_data) in enumerate(
            dataset[dataset["labels"] == class_label].iterrows()
        ):
            axes[img_idx, class_idx].imshow(img_data["images"])
            axes[img_idx, class_idx].axis("off")
            axes[img_idx, class_idx].set_title(f"img label: {class_label}")
            if img_idx + 1 >= n_images_per_class:
                break
    plt.show()


def plot_lowest_highest_influence_images(
    subset_influences: NDArray[np.float64],
    subset_images: List[JpegImageFile],
    num_to_plot: int,
):
    """Given a set of images and their influences over another, plots two columns
    of `num_to_plot` images each. Those on the right column have the lowest influence,
     those on the right the highest.

    Args:
        subset_influences: an array with influence values
        subset_images: a list of images
        num_to_plot: int, number of high and low influence images to plot
    """
    top_if_idxs = np.argsort(subset_influences)[-num_to_plot:]
    bottom_if_idxs = np.argsort(subset_influences)[:num_to_plot]

    fig, axes = plt.subplots(nrows=num_to_plot, ncols=2)
    fig.suptitle("Lowest (left) and highest (right) influences")

    for plt_idx, img_idx in enumerate(bottom_if_idxs):
        axes[plt_idx, 0].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 0].imshow(subset_images[img_idx])
        axes[plt_idx, 0].axis("off")

    for plt_idx, img_idx in enumerate(top_if_idxs):
        axes[plt_idx, 1].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 1].imshow(subset_images[img_idx])
        axes[plt_idx, 1].axis("off")

    plt.show()


def plot_losses(losses: Losses):
    """Plots the train and validation loss

    Args:
        losses: list of losses, one per epoch
    """
    _, ax = plt.subplots()
    ax.plot(losses.training, label="Train")
    ax.plot(losses.validation, label="Val")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Train epoch")
    ax.legend()
    plt.show()


def compute_mean_corrupted_influences(
    corrupted_dataset: pd.DataFrame,
    corrupted_indices: Dict[Any, List[int]],
    avg_corrupted_influences: NDArray[np.float64],
) -> pd.DataFrame:
    """Given a corrupted dataset, it returns a dataframe with average influence for each class,
    separating corrupted and original points.

    Args:
        corrupted_dataset: corrupted dataset as returned by get_corrupted_imagenet
        corrupted_indices: list of corrupted indices, as returned by get_corrupted_imagenet
        avg_corrupted_influences: average influence of each training point on the test dataset

    Returns:
        a dataframe holding the average influence of corrupted and non-corrupted data
    """
    labels = corrupted_dataset["labels"].unique()
    avg_label_influence = pd.DataFrame(
        columns=["label", "avg_non_corrupted_infl", "avg_corrupted_infl", "score_diff"]
    )
    for idx, label in enumerate(labels):
        avg_influences_series = pd.Series(avg_corrupted_influences)
        class_influences = avg_influences_series[corrupted_dataset["labels"] == label]
        corrupted_infl = class_influences[
            class_influences.index.isin(corrupted_indices[label])
        ]
        non_corrupted_infl = class_influences[
            ~class_influences.index.isin(corrupted_indices[label])
        ]
        avg_non_corrupted = np.mean(non_corrupted_infl)
        avg_corrupted = np.mean(corrupted_infl)
        avg_label_influence.loc[idx] = [
            label,
            avg_non_corrupted,
            avg_corrupted,
            avg_non_corrupted - avg_corrupted,
        ]
    return avg_label_influence


def plot_corrupted_influences_distribution(
    corrupted_dataset: pd.DataFrame,
    corrupted_indices: Dict[Any, List[int]],
    avg_corrupted_influences: NDArray[np.float64],
    figsize: Tuple[int, int] = (16, 8),
):
    """Given a corrupted dataset, plots the histogram with the distribution of
    influence values. This is done separately for each label: each has a plot
    where the distribution of the influence of non-corrupted points is compared
    to that of corrupted ones.

    Args:
        corrupted_dataset: corrupted dataset as returned by
            get_corrupted_imagenet
        corrupted_indices: list of corrupted indices, as returned by
            get_corrupted_imagenet
        avg_corrupted_influences: average influence of each training point on
            the test dataset
        figsize: for `plt.subplots()`

    Returns:
        a dataframe holding the average influence of corrupted and
        non-corrupted data
    """
    labels = corrupted_dataset["labels"].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("Distribution of corrupted and clean influences.")
    for idx, label in enumerate(labels):
        avg_influences_series = pd.Series(avg_corrupted_influences)
        class_influences = avg_influences_series[corrupted_dataset["labels"] == label]
        corrupted_infl = class_influences[
            class_influences.index.isin(corrupted_indices[label])
        ]
        non_corrupted_infl = class_influences[
            ~class_influences.index.isin(corrupted_indices[label])
        ]
        axes[idx].hist(non_corrupted_infl, label="Non corrupted", alpha=0.7)
        axes[idx].hist(corrupted_infl, label="Corrupted", alpha=0.7, color="green")
        axes[idx].set_xlabel("Influence values")
        axes[idx].set_ylabel("Number of samples")
        axes[idx].set_title(f"Influences for {label=}")
        axes[idx].legend()
    plt.show()


def filecache(path: Path | str) -> Callable[[Callable], Callable]:
    """Wraps a function to cache its output on disk.

    There is no hashing of the arguments of the function. This decorator merely
    checks whether `filename` exists and if so, loads the output from it, and if
    not it calls the function and saves the output to `filename`.

    The decorated function accepts two additional keyword arguments:

        _force_rebuild: if set to `True`, calls the wrapped function to recompute the
            output and overwrites the cached file.
        _silent: if `False`, prints messages about the cache status.

    Args:
        path: Path to the file to cache the output to.
    Returns:
        A wrapper function that caches the output of the wrapped function to the
            specified path.
    """
    if isinstance(path, str):
        path = Path(path)

    def decorator(fun: Callable) -> Callable:
        @wraps(fun)
        def wrapper(
            *args, _silent: bool = False, _force_rebuild: bool = False, **kwargs
        ) -> Any:
            try:
                with path.open("rb") as fd:
                    data = pickle.load(fd)
                    if not _silent:
                        print(f"Found cached file: {path.name}.")
                    if _force_rebuild:
                        if not _silent:
                            print("Ignoring and rebuilding...")
                        raise FileNotFoundError
                    return data
            except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
                if not isinstance(e, FileNotFoundError):
                    print(f"Unpickling error '{str(e)}'. Rebuilding '{path}'...")
                result = fun(*args, **kwargs)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as fd:
                    pickle.dump(result, fd)
                return result

        return wrapper

    return decorator


class ConstantBinaryClassifier(BaseEstimator):
    def __init__(self, p: float, random_state: int | None = None):
        """A classifier that always predicts class 0 with probability p and class 1
        with probability 1-p.

        The prediction is fixed upon fitting the model and constant for all
        outputs, i.e. the same prediction is returned for all samples. Call
        `fit()` again to resample the prediction.

        Args:
            p: probability that this estimator always predicts class 0
            random_state:
        """
        self.p = p
        self.prediction: int | None = None
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_outputs = y.shape[1] if y.ndim > 1 else 1
        self.prediction = (rng.random(n_outputs) > self.p).astype(int)

    def predict(self, X):
        return np.repeat(self.prediction, len(X))


class ThresholdTunerCV(BaseEstimator, ClassifierMixin):
    """
    A wrapper that tunes the decision threshold of a binary classifier to maximize
    a given metric, using cross-fitting on the training data.

    This is used to counteract class imbalance in the dataset. Note however that
    upsampling or downsampling the dataset can perform equally well or better, while
    being simpler to implement and faster to train.

    !!! Note
        This class is a left-over from a previous version of the Data-OOB notebook and
        should probably be removed.
    """

    def __init__(
        self,
        base_estimator,
        n_splits: int = 5,
        metric=f1_score,
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.metric = metric
        self.optimal_threshold_: float | None = None
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        # We find an optimal decision threshold using out-of-sample predictions
        cv_strategy = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        y_proba_cv = cross_val_predict(
            self.base_estimator,
            X,
            y,
            cv=cv_strategy,
            method="predict_proba",
            n_jobs=self.n_jobs,
        )[:, 1]

        thresholds = np.linspace(0, 1, 100)
        f1_scores = [self.metric(y, y_proba_cv >= t) for t in thresholds]
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Then we fit the model on the entire dataset
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        y_proba = self.base_estimator.predict_proba(X)[:, 1]
        return (y_proba >= self.optimal_threshold).astype(int)

    def __getattr__(self, item):
        return getattr(self.base_estimator, item)


if torch := try_torch_import():

    def num_model_params(model: torch.nn.Module) -> int:
        """Returns the number of parameters of a model.

        Args:
            model: A torch model.

        Returns:
            The number of parameters of the model.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def profile_torch_fit(
        model: skorch.NeuralNetClassifier, train: Dataset, test: Dataset, device: str
    ):
        """
        Profiles the fit of a torch model inside a skorch wrapper.

        Args:
            model: A skorch model.
            train: Training dataset
            test: Test set
            device: Device to use for the model.
        """
        import torch.autograd.profiler as profiler

        model.initialize_module()
        _model = model.module_
        _model.to(device=device)
        print(f"Model has {num_model_params(_model)} parameters")
        with profiler.profile(record_shapes=False, use_cuda=True) as prof:
            with profiler.record_function("model_forward"):
                model.fit(*train.data())
                print(f"Training accuracy: {model.score(*train.data()):.3f}")
                print(f"Test accuracy: {model.score(*test.data()):.3f}")
        torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
