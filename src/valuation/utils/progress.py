import collections

from tqdm.auto import tqdm, trange
from typing import Iterator, Literal, Union, overload


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


@overload
def maybe_progress(it: Union[int, Iterator], display: Literal[True],
                   **tqdm_kwargs) -> tqdm:
    pass


@overload
def maybe_progress(it: Union[int, Iterator], display: Literal[False],
                   **tqdm_kwargs) -> Iterator:
    pass


def maybe_progress(it: Union[int, Iterator], display: bool, **tqdm_kwargs)\
        -> Union[tqdm, Iterator]:
    """ Returns either a tqdm progress bar or a mock object which wraps the
    iterator as well, but ignores any accesses to methods or properties.
    :param it: the iterator to wrap
    :param display: set to True to return a tqdm bar
    :param **tqdm_kwargs: will be forwarded to tqdm
    """
    if isinstance(it, int):
        it = range(it)
        _tqdm = trange
    else:
        _tqdm = tqdm
    return _tqdm(it, **tqdm_kwargs) if display else MockProgress(it)
