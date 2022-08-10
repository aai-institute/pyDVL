from typing import Iterable, Iterator, Union

from tqdm.auto import tqdm

__all__ = ["maybe_progress"]


def maybe_progress(
    it: Union[int, Iterable, Iterator], display: bool = False, **kwargs
) -> Union[tqdm, Iterator]:
    """Returns either a tqdm progress bar or a mock object which wraps the
    iterator as well, but ignores any accesses to methods or properties.

    :param it: the iterator to wrap
    :param display: set to True to return a tqdm bar
    :param kwargs: Keyword arguments that will be forwarded to tqdm
    """
    if isinstance(it, int):
        it = range(it)
    return tqdm(it, **kwargs) if display else it
