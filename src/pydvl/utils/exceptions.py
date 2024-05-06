from functools import wraps
from typing import TypeVar, Type, Callable

CatchExceptionType = TypeVar("CatchExceptionType", bound=BaseException)
RaiseExceptionType = TypeVar("RaiseExceptionType", bound=BaseException)


def catch_and_raise_exception(
    catch_exception_type: Type[CatchExceptionType],
    raise_exception_factory: Callable[[CatchExceptionType], RaiseExceptionType],
) -> Callable:
    """
    A decorator that catches exceptions of a specified exception type and raises
    another specified exception.

    Args:
        catch_exception_type: The type of the exception to catch.
        raise_exception_factory: A factory function that creates a new exception.

    Returns:
        A decorator function that wraps the target function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except catch_exception_type as e:
                raise raise_exception_factory(e) from e

        return wrapper

    return decorator
