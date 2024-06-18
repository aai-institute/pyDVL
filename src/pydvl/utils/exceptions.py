from functools import wraps
from typing import Callable, Type, TypeVar


class NotFittedException(ValueError):
    def __init__(self, object_type: Type):
        super().__init__(
            f"Objects of type {object_type} must be fitted before calling "
            f"methods. "
            f"Call method fit with appropriate input."
        )


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

    ??? Example

        ```python
        @catch_and_raise_exception(RuntimeError, lambda e: TorchLinalgEighException(e))
        def safe_torch_linalg_eigh(*args, **kwargs):
            '''
            A wrapper around `torch.linalg.eigh` that safely handles potential runtime errors
            by raising a custom `TorchLinalgEighException` with more context,
            especially related to the issues reported in
            https://github.com/pytorch/pytorch/issues/92141.

            Args:
            *args: Positional arguments passed to `torch.linalg.eigh`.
            **kwargs: Keyword arguments passed to `torch.linalg.eigh`.

            Returns:
            The result of calling `torch.linalg.eigh` with the provided arguments.

            Raises:
            TorchLinalgEighException: If a `RuntimeError` occurs during the execution of
            `torch.linalg.eigh`.
            '''
            return torch.linalg.eigh(*args, **kwargs)
        ```
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
