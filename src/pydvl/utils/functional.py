"""
Supporting utilities for manipulating functions.
"""

from __future__ import annotations

import functools
import inspect
import time
import warnings
from logging import Logger, getLogger
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from weakref import WeakKeyDictionary

from typing_extensions import Concatenate, ParamSpec

__all__ = ["maybe_add_argument", "suppress_warnings", "timed"]


logger = getLogger(__name__)


def _accept_additional_argument(*args, fun: Callable, arg: str, **kwargs):
    """Calls the given function with the given positional and keyword arguments,
    removing `arg` from the keyword arguments.

    Args:
        args: Positional arguments to pass to the function.
        fun: The function to call.
        arg: The name of the argument to remove.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function.
    """
    try:
        del kwargs[arg]
    except KeyError:
        pass

    return fun(*args, **kwargs)


def free_arguments(fun: Union[Callable, functools.partial]) -> Set[str]:
    """Computes the set of free arguments for a function or
    [functools.partial][] object.

    All arguments of a function are considered free unless they are set by a
    partial. For example, if `f = partial(g, a=1)`, then `a` is not a free
    argument of `f`.

    Args:
        fun: A callable or a [partial object][].

    Returns:
        The set of free arguments of `fun`.

    !!! tip "New in version 0.7.0"
    """
    args_set_by_partial: Set[str] = set()

    def _rec_unroll_partial_function_args(
        g: Union[Callable, functools.partial],
    ) -> Callable:
        """Stores arguments and recursively call itself if `g` is a
        [functools.partial][] object. In the end, returns the initially wrapped
        function.

        This handles the construct `partial(_accept_additional_argument, *args,
        **kwargs)` that is used by `maybe_add_argument`.

        Args:
            g: A partial or a function to unroll.

        Returns:
            Initial wrapped function.
        """
        nonlocal args_set_by_partial

        if isinstance(g, functools.partial) and g.func == _accept_additional_argument:
            arg = g.keywords["arg"]
            if arg in args_set_by_partial:
                args_set_by_partial.remove(arg)
            return _rec_unroll_partial_function_args(g.keywords["fun"])
        elif isinstance(g, functools.partial):
            args_set_by_partial.update(g.keywords.keys())
            args_set_by_partial.update(g.args)
            return _rec_unroll_partial_function_args(g.func)
        else:
            return g

    wrapped_fn = _rec_unroll_partial_function_args(fun)
    sig = inspect.signature(wrapped_fn)
    return args_set_by_partial | set(sig.parameters.keys())


def maybe_add_argument(fun: Callable, new_arg: str) -> Callable:
    """Wraps a function to accept the given keyword parameter if it doesn't
    already.

    If `fun` already takes a keyword parameter of name `new_arg`, then it is
    returned as is. Otherwise, a wrapper is returned which merely ignores the
    argument.

    Args:
        fun: The function to wrap
        new_arg: The name of the argument that the new function will accept
            (and ignore).

    Returns:
        A new function accepting one more keyword argument.

    !!! tip "Changed in version 0.7.0"
        Ability to work with partials.
    """
    if new_arg in free_arguments(fun):
        return fun

    return functools.partial(_accept_additional_argument, fun=fun, arg=new_arg)


F = TypeVar("F", bound=Callable[..., Any])


class WarningsSuppressor:
    """A descriptor-based decorator to conditionally suppress warnings of a given
    set of categories when the instance attribute (specified by `flag_attribute`) is
    `False`.

    It ensures that:
      - The decorated function is an instance method (the first parameter must be
        `self`).
      - The instance possesses the attribute named by `flag_attribute` at call time.
    """

    def __init__(
        self, func: F, categories: Sequence[Type[Warning]], flag_attribute: str
    ) -> None:
        functools.update_wrapper(cast(F, self), func)
        self.func = func
        self.categories = categories
        self.flag_attribute = flag_attribute

        # HACK: Crappy heuristic to verify that the function is an instance method
        #  At decoration time it's not yet bound, so we must resort to this sort of crap
        sig = inspect.signature(func)
        params = list(sig.parameters)
        if not params or params[0] != "self":
            raise TypeError(
                "suppress_warnings decorator can only be applied to instance methods "
                "(first parameter must be 'self')."
            )

    def __get__(
        self, instance: Any, owner: Optional[Type[Any]] = None
    ) -> Callable[..., Any]:
        if instance is None:
            return cast(Callable[..., Any], self)  # Allow access via the class.

        @functools.wraps(self.func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if getattr(instance, self.flag_attribute, False):
                return self.func(instance, *args, **kwargs)
            with warnings.catch_warnings():
                for category in self.categories:
                    warnings.simplefilter("ignore", category=category)
                return self.func(instance, *args, **kwargs)

        return wrapper


def suppress_warnings(
    categories: Sequence[Type[Warning]] = (Warning,), flag: str = "show_warnings"
) -> Callable[[F], F]:
    """
    A decorator to suppress warnings in class methods.

    ??? Example "Suppress all warnings"
        ```python
        class A:
            @suppress_warnings()
            def method(self, ...):
                ...
        ```
    ??? Example "Suppress only `UserWarning`
        ```python
        class A:
            def __init__(self, show_warnings: bool):
                # the decorator will look for this attribute by default
                self.show_warnings = show_warnings

            @suppress_warnings(categories=(UserWarning,))
            def method(self, ...):
                ...
        ```
    ??? Example "Configuring behaviour at runtime"
        ```python
        class A:
            def __init__(self, warn_enabled: bool)
                self.warn_enabled = warn_enabled

            @suppress_warnings(flag="warn_enabled")
            def method(self, ...):
                ....
        ```
    """

    def wrapper(func: F) -> F:
        return cast(
            F, WarningsSuppressor(func, categories=categories, flag_attribute=flag)
        )

    return wrapper


P = ParamSpec("P")  # Args of fun
R = TypeVar("R")  # Return type
T = TypeVar("T")  # Instance type for methods


class Timed(Generic[T, P, R]):
    """A decorator that measures the execution time of the wrapped function, storing
     the elapsed time in the 'execution_time' attribute.

    The decorator preserves the metadata of the original function and supports both
    plain functions and instance methods.

    To decorate functions, you probably want to call `timed` instead.

    Attributes:
        execution_time (float): The time taken by the last call to the decorated
            function in seconds.
    Args:
        fun (Callable): The function to decorate.
        accumulate (bool): If `True`, the execution time will be accumulated across
            multiple calls. Otherwise, it will be reset on each call.
        logger (Optional[Logger]): If provided, log the execution time at the INFO
            level.
    """

    def __init__(
        self,
        fun: Union[Callable[Concatenate[T, P], R], Callable[P, R]],
        accumulate: bool = False,
        logger: Logger | None = None,
    ) -> None:
        self.execution_time = 0.0
        self.fun = fun
        self.accumulate = accumulate
        self.logger: Logger | None = logger
        self._method_cache: WeakKeyDictionary[Any, Timed[T, P, R]] = WeakKeyDictionary()
        functools.update_wrapper(self, fun)

    # For a plain callable (or a bound method)
    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    # For an unbound method (accessed via the class: there is a self argument)
    @overload
    def __call__(self, instance: T, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = self.fun(*args, **kwargs)
        finally:
            t = time.perf_counter() - start
            self.execution_time = self.execution_time + t if self.accumulate else t
        if self.logger is not None:
            self.logger.log(
                self.logger.level,
                f"{getattr(self.fun, '__objclass__', None) or self.fun.__module__}."
                f"{self.fun.__qualname__} took {self.execution_time:.5f} seconds",
            )
        return result

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Supports instance methods by binding the wrapped function to the instance.
        When accessed as a method, returns a `Timed` object around the bound function,
        or one from cache if it was already bound (the cache is necessary in order to be
        able to access the `execution_time` attribute, otherwise we would create a new
        object on each access).
        """
        if instance is None:  # Allow access to the descriptor itself
            return self

        if instance not in self._method_cache:
            bound_fun: Callable[P, R] = self.fun.__get__(instance, owner)
            new_wrapper: Timed[T, P, R] = Timed(
                bound_fun, accumulate=self.accumulate, logger=self.logger
            )
            self._method_cache[instance] = new_wrapper
        return self._method_cache[instance]

    def __repr__(self) -> str:
        return f"<Timed({self.fun.__name__}) at {hex(id(self))}>"

    def reset(self) -> None:
        """Reset the execution time to 0. Useful only when the timer was initialised
        with `accumulate=True`."""
        self.execution_time = 0.0


def timed(
    *, accumulate: bool = False, logger: Logger | None = None
) -> Callable[[Callable[..., R]], Timed[Any, P, R]]:
    """A decorator that measures the execution time of the wrapped function, storing
    the elapsed time in the 'execution_time' attribute.

    The decorator preserves the metadata of the original function and supports both
    plain functions and instance methods.

    Args:
        fun: The function to decorate.
        accumulate: If `True`, the execution time will be accumulated across multiple
            calls. Otherwise, it will be reset on each call.
        logger: If provided, log the execution time at the INFO level.

    Returns:
        A [Timed][pydvl.utils.functional.Timed] object wrapping the decorated function.
    """

    def wrapper(fun: Callable[..., R]) -> Timed[Any, P, R]:
        return Timed(fun, accumulate=accumulate, logger=logger)

    return wrapper
