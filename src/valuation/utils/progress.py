import collections

from tqdm.auto import tqdm
from typing import Iterator


class MockProgress(collections.Iterator):
    """ A Naive mock class to use with maybe_progress and tqdm.

     Mocked methods don't support return values.
     Mocked properties don't do anything
     """

    class MiniMock:
        def __call__(self, *args, **kwargs): pass
        def __add__(self, other): pass
        def __sub__(self, other): pass
        def __mul__(self, other): pass
        def __floordiv__(self, other): pass
        def __truediv__(self, other): pass

    def __init__(self, iterator: Iterator):
        # Since there is no _it in __dict__ at this point, doing here
        # self._it = iterator
        # results in a call to __getattr__() and the assignment fails, so we
        # use __dict__ instead
        self.__dict__['_it'] = iterator

    def __iter__(self):
        return iter(self._it)

    def __next__(self):
        return next(self._it)

    def __getattr__(self, key):
        return self.MiniMock()

    def __setattr__(self, key, value):
        pass


def maybe_progress(it: Iterator, display: bool, **tqdm_kwargs):
    """ Returns either a tqdm progress bar or a mock object which wraps the
    iterator as well, but ignores any accesses to methods or properties.
    :param it: the iterator to wrap
    :param display: set to True to return a tqdm bar
    :param **tqdm_kwargs: will be forwarded to tqdm
    """
    return tqdm(it, **tqdm_kwargs) if display else MockProgress(it)
