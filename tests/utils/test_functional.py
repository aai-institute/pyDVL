from __future__ import annotations

import inspect
import logging
import time
import warnings
from time import sleep
from typing import Any, Type

import numpy as np
import pytest

from pydvl.utils.functional import suppress_warnings, timed


class WarningsClass:
    def __init__(self, show_warnings: str | bool = True):
        self.show_warnings = show_warnings

    @suppress_warnings(categories=(UserWarning,), flag="show_warnings")
    def method_warn(self) -> str:
        warnings.warn("User warning", UserWarning)
        return "done"

    @suppress_warnings(categories=(DeprecationWarning,), flag="show_warnings")
    def method_deprecation(self) -> str:
        warnings.warn("Deprecated", DeprecationWarning)
        return "done"

    @suppress_warnings(categories=(RuntimeWarning,), flag="show_warnings")
    def division_by_zero(self) -> float:
        # This will trigger a RuntimeWarning from numpy.
        return np.log(0)

    @suppress_warnings
    def runtime_warning_no_explicit_category(self) -> float:
        return np.log(0)


def test_warning_shown_method(recwarn: Any):
    obj = WarningsClass(show_warnings=True)
    result = obj.method_warn()
    assert result == "done"

    # recwarn captures warnings issued during the test.
    w = recwarn.pop(UserWarning)
    assert "User warning" in str(w.message)
    assert not recwarn  # Ensure no extra warnings were issued.


def test_warning_suppressed_method():
    obj = WarningsClass(show_warnings=False)
    with warnings.catch_warnings(record=True) as record:
        result = obj.method_warn()
        assert result == "done"
        assert len(record) == 0


def test_warning_suppressed_function(recwarn: Any):
    def fun():
        warnings.warn("User warning", UserWarning)

    silent_fun = suppress_warnings(fun)
    with warnings.catch_warnings(record=True) as record:
        silent_fun()
        assert len(record) == 0


def test_warning_noflag_for_methods():
    def fun():
        warnings.warn("User warning", UserWarning)

    with pytest.raises(ValueError):
        suppress_warnings(fun, flag="whatever")


def test_any_warning_suppressed():
    obj = WarningsClass(show_warnings=False)
    with warnings.catch_warnings(record=True) as record:
        result = obj.runtime_warning_no_explicit_category()
        assert result == -np.inf
        assert len(record) == 0


@pytest.mark.parametrize(
    "method_name, category, warning_message",
    [
        ("method_warn", UserWarning, "User warning"),
        ("method_deprecation", DeprecationWarning, "Deprecated"),
    ],
)
def test_different_categories(
    method_name: str, category: Type[Warning], warning_message: str, recwarn: Any
):
    obj = WarningsClass(show_warnings=True)
    method = getattr(obj, method_name)
    result = method()
    assert result == "done"
    w = recwarn.pop(category)
    assert warning_message in str(w.message)


def test_raises_on_flag_error():
    obj = WarningsClass(show_warnings="error")
    with pytest.raises(UserWarning):
        obj.method_warn()


def test_invalid_flag_type():
    obj = WarningsClass(show_warnings=42)
    with pytest.raises(TypeError):
        obj.method_warn()


def test_nonmethod_decorator_usage():
    @suppress_warnings(categories=(RuntimeWarning,))
    def fun(x: int) -> float:
        return np.log(x)

    with warnings.catch_warnings(record=True) as record:
        result = fun(0)
        assert result == -np.inf
        assert len(record) == 0


class MultiWarningClass:
    def __init__(self, warn_flag: bool = True):
        self.warn_flag = warn_flag

    @suppress_warnings(categories=(UserWarning, DeprecationWarning), flag="warn_flag")
    def multi_warning(self) -> str:
        warnings.warn("User warning", UserWarning)
        warnings.warn("Deprecated", DeprecationWarning)
        return "done"


def test_multi_warning_shown(recwarn: Any):
    # With the custom flag True, both warnings should be issued.
    obj = MultiWarningClass(warn_flag=True)
    result = obj.multi_warning()
    assert result == "done"
    w1 = recwarn.pop(UserWarning)
    assert "User warning" in str(w1.message)
    w2 = recwarn.pop(DeprecationWarning)
    assert "Deprecated" in str(w2.message)
    assert not recwarn


def test_multi_warning_suppressed():
    # With the custom flag False, both warnings should be suppressed.
    obj = MultiWarningClass(warn_flag=False)
    with warnings.catch_warnings(record=True) as record:
        result = obj.multi_warning()
        assert result == "done"
        # No warnings should be recorded.
        assert len(record) == 0


def test_timed_decorator_execution_time():
    """Test that the timed decorator correctly measures execution time"""

    @timed()
    def slow_function(sleep_time: float) -> None:
        sleep(sleep_time)

    sleep_duration = 0.1
    slow_function(sleep_duration)
    assert slow_function.execution_time >= sleep_duration


def test_timed_decorator_return_value():
    """Test that the timed decorator preserves the return value"""

    @timed()
    def identity(x: int) -> int:
        return x

    assert identity(42) == 42
    assert hasattr(identity, "execution_time")


def test_timed_decorator_with_args_kwargs():
    """Test that the timed decorator works with different argument patterns"""

    @timed()
    def function_with_args(*args, **kwargs) -> tuple:
        return args, kwargs

    args = (1, 2, 3)
    kwargs = {"a": 1, "b": 2}
    result_args, result_kwargs = function_with_args(*args, **kwargs)

    assert result_args == args
    assert result_kwargs == kwargs
    assert hasattr(function_with_args, "execution_time")


def test_timed_decorator_resets_time():
    """Test that each call updates the execution time"""

    @timed()
    def fun(fast: bool) -> None:
        if fast:
            return
        sleep(0.2)

    fun(fast=True)
    fast_time = fun.execution_time

    fun(fast=False)
    slow_time = fun.execution_time

    assert slow_time > fast_time


#######################################################


@pytest.fixture
def timed_function():
    @timed
    def fun(arg: int, kwarg: int = 0) -> int:
        time.sleep(0.01)
        return arg + kwarg

    return fun


@pytest.fixture
def timed_class():
    class TestClass:
        @timed  # no args form
        def method(self, arg: int) -> int:
            time.sleep(0.01)
            return arg * 2

        @timed(accumulate=True)
        def accumulating_method(self, arg: int) -> int:
            time.sleep(0.01)
            return arg * 2

        @timed()  # explicit no args form
        def raises_exception(self):
            time.sleep(0.01)
            raise ValueError("Intentional error")

    return TestClass


def test_function_timing(timed_function):
    result = timed_function(2, kwarg=3)
    assert result == 5
    assert timed_function.execution_time >= 0.01


def test_method_timing(timed_class):
    obj = timed_class()
    result = obj.method(5)
    assert result == 10
    assert obj.method.execution_time >= 0.01


@pytest.mark.flaky(reruns=1)
def test_consecutive_calls(timed_function):
    times = []
    for _ in range(10):
        timed_function(1)
        times.append(timed_function.execution_time)

    np.testing.assert_allclose(np.mean(times), 0.01, rtol=0.1)


def test_accumulating_method(timed_class):
    obj = timed_class()

    initial_time = obj.accumulating_method.execution_time
    assert initial_time == 0.0

    obj.accumulating_method(1)
    time_after_first_call = obj.accumulating_method.execution_time
    assert time_after_first_call > initial_time

    obj.accumulating_method(1)
    time_after_second_call = obj.accumulating_method.execution_time
    assert time_after_second_call > time_after_first_call


def test_function_metadata(timed_function):
    assert timed_function.__name__ == "fun"
    signature = str(inspect.signature(timed_function))
    assert "arg" in signature
    assert "int" in signature


def test_method_metadata(timed_class):
    obj = timed_class()
    method = obj.method
    assert method.__name__ == "method"
    signature = str(inspect.signature(method))
    assert "arg" in signature
    assert "int" in signature


def test_separate_instance_timing(timed_class):
    obj1 = timed_class()
    obj2 = timed_class()

    obj1.method(1)
    obj2.method(1)

    assert obj1.method.execution_time >= 0.01
    assert obj2.method.execution_time >= 0.01
    np.testing.assert_allclose(
        obj1.method.execution_time, obj2.method.execution_time, rtol=0.1
    )


def test_exception_propagation(timed_class):
    obj = timed_class()
    with pytest.raises(ValueError):
        obj.raises_exception()


def test_exception_timing(timed_class):
    obj = timed_class()
    try:
        obj.raises_exception()
    except ValueError:
        pass
    assert obj.raises_exception.execution_time >= 0.01


def test_descriptor_access(timed_class):
    decorator = timed_class.__dict__["method"]
    assert decorator is timed_class.method


def test_logging_output(caplog):
    @timed(logger=logging.getLogger("test"))
    def logging_fun(arg: int, kwarg: int = 0) -> int:
        time.sleep(0.01)
        return arg + kwarg

    with caplog.at_level(level=logging.INFO, logger="test"):
        logging_fun(1)
        assert any(record.msg.endswith("seconds") for record in caplog.records)


def test_method_logging_output(caplog):
    class TestClass:
        @timed(logger=logging.getLogger("test"))
        def method(self, arg: int) -> int:
            time.sleep(0.01)
            return arg * 2

    obj = TestClass()
    with caplog.at_level(level=logging.INFO, logger="test"):
        obj.method(1)
        assert any(record.msg.endswith("seconds") for record in caplog.records)


def test_zero_execution_time(timed_function):
    @timed()
    def instant_func():
        return

    instant_func()
    assert instant_func.execution_time < 0.01


def test_accumulated_time():
    @timed(accumulate=True)
    def slow_function():
        time.sleep(0.01)

    slow_function()
    slow_function()
    assert slow_function.execution_time >= 0.02


def test_method_accumulated_time():
    class TestClass:
        @timed(accumulate=True)
        def method(self):
            time.sleep(0.01)

    obj = TestClass()
    obj.method()
    obj.method()
    assert obj.method.execution_time >= 0.02


def test_return_type_preservation(timed_function):
    result = timed_function(1)
    assert isinstance(result, int)


def test_method_return_type_preservation(timed_class):
    result = timed_class().method(2)
    assert isinstance(result, int)
