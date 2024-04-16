from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from joblib._parallel_backends import (
    LokyBackend,
    MultiprocessingBackend,
    ThreadingBackend,
)
from joblib.parallel import _get_active_backend
from multiprocessing import shared_memory


def ensure_backend_has_generator_return():
    backend = _get_active_backend()[0]
    if not backend.supports_return_generator:
        raise ValueError(
            "The current backend does not support generator return. "
            "Please use a different backend. If you are using ray, follow the "
            "instructions in the documentation to enable generator support."
        )
        # TODO: copy this to the docs:
        #
        #  Instead of calling register_ray(), Do this to enable generator support in
        #  Ray. See:
        #  https://github.com/aai-institute/pyDVL/issues/387#issuecomment-1807994301
        #
        # from ray.util.joblib.ray_backend import RayBackend
        # RayBackend.supports_return_generator = True
        # register_parallel_backend("ray", RayBackend)


class Flag(ABC):
    @abstractmethod
    def set(self):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def __call__(self):
        ...

    def __bool__(self):  # some syntactic sugar
        return self.__call__()

    @abstractmethod
    def unlink(self):
        ...


class ThreadingFlag(Flag):
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def reset(self):
        self._flag = False

    def __call__(self):
        return self._flag

    def unlink(self):
        pass


class MultiprocessingFlag(Flag):
    def __init__(self, name: str):
        self._flag = shared_memory.SharedMemory(name, create=False, size=1)

    def set(self):
        self._flag.buf[0] = True

    def reset(self):
        self._flag.buf[0] = False

    def __call__(self):
        return self._flag.buf[0]

    @classmethod
    def create(cls):
        uid = uuid.uuid4().hex
        shared_memory.SharedMemory(uid, create=True, size=1)
        return cls(uid)

    def unlink(self):
        self._flag.close()
        self._flag.unlink()


from contextlib import contextmanager


@contextmanager
def make_parallel_flag():
    backend = _get_active_backend()[0]

    if isinstance(backend, MultiprocessingBackend) or isinstance(backend, LokyBackend):
        flag = MultiprocessingFlag.create()
    elif isinstance(backend, ThreadingBackend):
        flag = ThreadingFlag()
    else:
        raise NotImplementedError()

    try:
        yield flag
    finally:
        flag.unlink()
