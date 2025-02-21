import warnings
from typing import Any, Type

import numpy as np
import pytest

from pydvl.utils.functional import suppress_warnings


class WarningsClass:
    def __init__(self, show_warnings: bool = True):
        self.show_warnings = show_warnings

    @suppress_warnings(categories=(UserWarning,))
    def method_warn(self) -> str:
        warnings.warn("User warning", UserWarning)
        return "done"

    @suppress_warnings(categories=(DeprecationWarning,))
    def method_deprecation(self) -> str:
        warnings.warn("Deprecated", DeprecationWarning)
        return "done"

    @suppress_warnings(categories=(RuntimeWarning,))
    def division_by_zero(self) -> float:
        # This will trigger a RuntimeWarning from numpy.
        return np.log(0)

    @suppress_warnings()
    def runtime_warning_no_explicit_category(self) -> float:
        return np.log(0)


def test_warning_shown(recwarn: Any):
    # When show_warnings is True, the warning should be emitted.
    obj = WarningsClass(show_warnings=True)
    result = obj.method_warn()
    assert result == "done"

    # recwarn captures warnings issued during the test.
    w = recwarn.pop(UserWarning)
    assert "User warning" in str(w.message)
    assert not recwarn  # Ensure no extra warnings were issued.


def test_warning_suppressed():
    # When show_warnings is False, the warning should be suppressed.
    obj = WarningsClass(show_warnings=False)
    with warnings.catch_warnings(record=True) as record:
        result = obj.method_warn()
        assert result == "done"
        assert len(record) == 0


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


def test_invalid_decorator_usage():
    # Applying the decorator to a function that is not an instance method should raise a TypeError.
    with pytest.raises(TypeError):

        @suppress_warnings(categories=(UserWarning,))
        def not_a_method(x: int) -> int:
            return x


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
