from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Type, Union

from pymemcache.serde import PickleSerde

from .types import unpackable

PICKLE_VERSION = 5  # python >= 3.8

__all__ = ["MemcachedClientConfig", "MemcachedConfig"]


@unpackable
@dataclass
class MemcachedClientConfig:
    server: Union[str, Tuple[str, Union[str, int]]] = ("localhost", 11211)
    connect_timeout: float = 1.0
    timeout: float = 1.0
    no_delay: bool = True
    serde: PickleSerde = PickleSerde(pickle_version=PICKLE_VERSION)


@unpackable
@dataclass
class MemcachedConfig:
    """Configuration for memcache

    :param cache_threshold: determines the minimum number of seconds a model training needs
        to take to cache its scores. If a model is super fast to train, you may just want
        to re-train it every time without saving the score. In most cases, caching the model,
        even when it takes very little to train, is preferable.
        The default to cache_threshold is 0.3 seconds.
    :param allow_repeated_training: if set to true, instead of storing just a single score of a model,
        the cache will store a running average of its score until a certain relative tolerance
        (set by the rtol_threshold argument) is achieved. More precisely, since most machine learning
        model-trainings are non-deterministic, depending on the starting weights or on randomness in
        the training process, the trained model can have very different scores.
        In your workflow, if you observe that the training process is very noisy even relative to the
        same training set, then we recommend to set allow_repeated_training to True.
        If instead the score is not impacted too much by non-deterministic training, setting allow_repeated_training
        to false will speed up the shapley_dval calculation substantially.
    :param rtol_threshold argument: as mentioned above, it regulates the relative tolerance for returning the running
        average of a model instead of re-training it. If allow_repeated_training is True, set rtol_threshold to
        small values and the shapley coefficients will have higher precision.
    :param min_repetitions: similarly to rtol_threshold, it regulates repeated trainings by setting the minimum number of
        repeated training a model has to go through before the cache can return its average score.
        If the model training is very noisy, set min_repetitions to higher values and the scores will be more
        reflective of the real average performance of the trained models.
    """

    client_config: MemcachedClientConfig = field(default_factory=MemcachedClientConfig)
    cache_threshold: float = 0.3
    allow_repeated_training: bool = True
    rtol_threshold: float = 0.1
    min_repetitions: int = 3
    ignore_args: Optional[Iterable[str]] = None
