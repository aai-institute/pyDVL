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
    Protocol,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import ParamSpec

__all__ = ["maybe_add_argument", "suppress_warnings", "timed"]


logger = getLogger(__name__)

R = TypeVar("R", covariant=True)


def _accept_additional_argument(
    *args: Any, fun: Callable[..., R], arg: str, **kwargs: Any
) -> R:
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
    """Computes the set of free arguments for a function or [[functools.partial]]
    object.

    All arguments of a function are considered free unless they are set by a
    partial. For example, if `f = partial(g, a=1)`, then `a` is not a free
    argument of `f`.

    Args:
        fun: A callable or a [partial object][functools.partial].

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


P = ParamSpec("P")


@overload
def suppress_warnings(fun: Callable[P, R]) -> Callable[P, R]: ...


@overload
def suppress_warnings(
    fun: None = None,
    *,
    categories: Sequence[Type[Warning]] = (Warning,),
    flag: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def suppress_warnings(
    fun: Callable[P, R],
    *,
    categories: Sequence[Type[Warning]] = (Warning,),
    flag: str = "",
) -> Callable[P, R]: ...


def suppress_warnings(
    fun: Callable[P, R] | None = None,
    *,
    categories: Sequence[Type[Warning]] = (Warning,),
    flag: str = "",
) -> Union[Callable[[Callable[P, R]], Callable[P, R]], Callable[P, R]]:
    """Decorator for class methods to conditionally suppress warnings.

      The decorated method will execute with warnings suppressed for the specified
      categories. If the instance has the attribute named by `flag`, and it's a boolean
      evaluating to `False`, warnings will be ignored. If the attribute is a string, then
      it is interpreted as an "action" to be performed on the categories specified.
      Allowed values are as per [warnings.simplefilter][], which are:
    `default`, `error`, `ignore`, `always`, `all`, `module`, `once`

      ??? Example "Suppress all warnings"
          ```python
          class A:
              @suppress_warnings
              def method(self, ...):
                  ...
          ```
      ??? Example "Suppress only `UserWarning`"
          ```python
          class A:
              @suppress_warnings(categories=(UserWarning,))
              def method(self, ...):
                  ...
          ```
      ??? Example "Configuring behaviour at runtime"
          ```python
          class A:
              def __init__(self, warn_enabled: bool):
                  self.warn_enabled = warn_enabled

              @suppress_warnings(flag="warn_enabled")
              def method(self, ...):
                  ...
          ```

      ??? Example "Raising on RuntimeWarning"
          ```python
          class A:
              def __init__(self, warnings: str = "error"):
                  self.warnings = warnings

              @suppress_warnings(flag="warnings")
              def method(self, ...):
                  ...

          A().method()  # Raises RuntimeWarning
          ```


      Args:
          fun: Optional callable to decorate. If provided, the decorator is applied inline.
          categories: Sequence of warning categories to suppress.
          flag: Name of an instance attribute to check for enabling warnings. If the
                attribute exists and evaluates to `False`, warnings will be ignored. If
                it evaluates to a str, then this action will be performed on the categories
                specified. Allowed values are as per [warnings.simplefilter][], which are:
                `default`, `error`, `ignore`, `always`, `all`, `module`, `once`

      Returns:
          Either a decorator (if no function is provided) or the decorated callable.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        # Use a simple heuristic: if the first parameter is "self", assume it's a method.
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        if not params or params[0] != "self":
            if flag:
                raise ValueError("Cannot use suppress_warnings flag with non-methods")

            @functools.wraps(fn)
            def suppress_warnings_wrapper(*args: Any, **kwargs: Any) -> R:
                with warnings.catch_warnings():
                    for category in categories:
                        warnings.simplefilter("ignore", category=category)
                    return fn(*args, **kwargs)

            return cast(Callable[P, R], suppress_warnings_wrapper)
        else:

            @functools.wraps(fn)
            def suppress_warnings_wrapper(self, *args: Any, **kwargs: Any) -> R:
                if flag and not hasattr(self, flag):
                    raise AttributeError(
                        f"Instance has no attribute '{flag}' for suppress_warnings"
                    )
                if flag and getattr(self, flag, False) is True:
                    return fn(self, *args, **kwargs)
                # flag is either False or a string
                with warnings.catch_warnings():
                    if (action := getattr(self, flag, "ignore")) is False:
                        action = "ignore"
                    elif not isinstance(action, str):
                        raise TypeError(
                            f"Expected a boolean or string for flag '{flag}', got {type(action).__name__}"
                        )
                    for category in categories:
                        warnings.simplefilter(action, category=category)  # type: ignore
                    return fn(self, *args, **kwargs)

            return cast(Callable[P, R], suppress_warnings_wrapper)

    if fun is None:
        return decorator
    return decorator(fun)


class TimedCallable(Protocol[P, R]):
    execution_time: float

    def __call__(self, *args, **kwargs) -> R: ...


@overload
def timed(fun: Callable[P, R]) -> TimedCallable[P, R]: ...


@overload
def timed(
    fun: None = None, *, accumulate: bool = False, logger: Logger | None = None
) -> Callable[[Callable[P, R]], TimedCallable[P, R]]: ...


@overload
def timed(
    fun: Callable[P, R], *, accumulate: bool = False, logger: Logger | None = None
) -> TimedCallable[P, R]: ...


def timed(
    fun: Callable[P, R] | None = None,
    *,
    accumulate: bool = False,
    logger: Logger | None = None,
) -> Union[Callable[[Callable[P, R]], TimedCallable[P, R]], TimedCallable[P, R]]:
    """A decorator that measures the execution time of the wrapped function.
    Optionally logs the time taken.

    ??? Example "Decorator usage"
        ```python
        @timed
        def fun(...):
            ...

        @timed(accumulate=True, logger=getLogger(__name__))
        def fun(...):
            ...
        ```

    ??? Example "Inline usage"
        ```python
        timed_fun = timed(fun)
        fun(...)
        print(timed_fun.execution_time)

        accum_time = timed(fun, accumulate=True)
        fun(...)
        fun(...)
        print(accum_time.execution_time)
        ```

    Args:
        fun:
        accumulate: If `True`, the total execution time will be accumulated across all
            calls.
        logger: If provided, the execution time will be logged at the logger's level.

    Returns:
        A decorator that wraps a function, measuring and optionally logging its
        execution time. The function will have an attribute `execution_time` where
        either the time of the last execution or the accumulated total is stored.
    """

    if fun is None:

        def decorator(func: Callable[P, R]) -> TimedCallable[P, R]:
            return timed(func, accumulate=accumulate, logger=logger)

        return decorator

    assert fun is not None

    @functools.wraps(fun)
    def timed_wrapper(*args, **kwargs) -> R:
        start = time.perf_counter()
        try:
            assert fun is not None
            result = fun(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            if accumulate:
                cast(TimedCallable, timed_wrapper).execution_time += elapsed
            else:
                cast(TimedCallable, timed_wrapper).execution_time = elapsed
            if logger is not None:
                assert fun is not None
                logger.log(
                    logger.level,
                    f"{fun.__module__}.{fun.__qualname__} took {elapsed:.5f} seconds",
                )
        return result

    cast(TimedCallable, timed_wrapper).execution_time = 0.0

    return cast(TimedCallable[P, R], timed_wrapper)
