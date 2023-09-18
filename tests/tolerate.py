import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence, Type

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.terminal import TerminalReporter

__all__ = ["TolerateErrorFixture", "TolerateErrorsSession", "wrap_pytest_function"]

EXCEPTIONS_TYPE = Optional[Sequence[Type[BaseException]]]


class TolerateErrorsSession:
    def __init__(self, config: "Config") -> None:
        self.verbose = config.getoption("tolerate_verbose")
        self.quiet = False if self.verbose else config.getoption("tolerate_quiet")
        self.columns = ["passed", "failed", "skipped", "max_failures"]
        self.labels = {
            "name": "Name",
            "passed": "Passed",
            "failed": "Failed",
            "skipped": "Skipped",
            "max_failures": "Maximum Allowed # Failures",
        }
        self._tests = defaultdict(TolerateErrorsTestItem)

    def get_max_failures(self, key: str) -> int:
        return self._tests[key].max_failures

    def set_max_failures(self, key: str, value: int) -> None:
        self._tests[key].max_failures = value

    def get_num_passed(self, key: str) -> int:
        return self._tests[key].passed

    def increment_num_passed(self, key: str) -> None:
        self._tests[key].passed += 1

    def get_num_failures(self, key: str) -> int:
        return self._tests[key].failed

    def increment_num_failures(self, key: str) -> None:
        self._tests[key].failed += 1

    def get_num_skipped(self, key: str) -> int:
        return self._tests[key].skipped

    def increment_num_skipped(self, key: str) -> None:
        self._tests[key].skipped += 1

    def set_exceptions_to_ignore(self, key: str, value: EXCEPTIONS_TYPE) -> None:
        if value is None:
            self._tests[key].exceptions_to_ignore = tuple()
        elif isinstance(value, Sequence):
            self._tests[key].exceptions_to_ignore = value
        else:
            self._tests[key].exceptions_to_ignore = (value,)

    def get_exceptions_to_ignore(self, key: str) -> EXCEPTIONS_TYPE:
        return self._tests[key].exceptions_to_ignore

    def has_exceeded_max_failures(self, key: str) -> bool:
        return self._tests[key].failed > self._tests[key].max_failures

    def display(self, terminalreporter: "TerminalReporter"):
        if self.quiet:
            return
        if len(self._tests) == 0:
            return
        terminalreporter.ensure_newline()
        terminalreporter.write_line("")
        widths = {
            "name": 3
            + max(len(self.labels["name"]), max(len(name) for name in self._tests))
        }
        for key in self.columns:
            widths[key] = 5 + len(self.labels[key])

        labels_line = self.labels["name"].ljust(widths["name"]) + "".join(
            self.labels[prop].rjust(widths[prop]) for prop in self.columns
        )
        terminalreporter.write_line(
            " tolerate: {count} tests ".format(count=len(self._tests)).center(
                len(labels_line), "-"
            ),
            yellow=True,
        )
        terminalreporter.write_line(labels_line)
        terminalreporter.write_line("-" * len(labels_line), yellow=True)
        for name in self._tests:
            has_error = self.has_exceeded_max_failures(name)
            terminalreporter.write(
                name.ljust(widths["name"]),
                red=has_error,
                green=not has_error,
                bold=True,
            )
            for prop in self.columns:
                terminalreporter.write(
                    "{0:>{1}}".format(self._tests[name][prop], widths[prop])
                )
            terminalreporter.write("\n")
        terminalreporter.write_line("-" * len(labels_line), yellow=True)
        terminalreporter.write_line("")


class TolerateErrorsTestItem:
    def __init__(self):
        self.max_failures = 0
        self.failed = 0
        self.passed = 0
        self.skipped = 0
        self.exceptions_to_ignore = tuple()

    def __getitem__(self, item: str):
        return getattr(self, item)


class TolerateErrorFixture:
    def __init__(self, node: pytest.Item):
        if hasattr(node, "originalname"):
            self.name = node.originalname
        else:
            self.name = node.name
        self.session: TolerateErrorsSession = node.config._tolerate_session
        marker = node.get_closest_marker("tolerate")
        if marker:
            max_failures = marker.kwargs.get("max_failures")
            exceptions_to_ignore = marker.kwargs.get("exceptions_to_ignore")
            self.session.set_max_failures(self.name, max_failures)
            self.session.set_exceptions_to_ignore(self.name, exceptions_to_ignore)

    def __call__(
        self, max_failures: int, *, exceptions_to_ignore: EXCEPTIONS_TYPE = None
    ):
        self.session.set_max_failures(self.name, max_failures)
        self.session.set_exceptions_to_ignore(self.name, exceptions_to_ignore)
        return self

    def __enter__(self):
        if self.session.has_exceeded_max_failures(self.name):
            self.session.increment_num_skipped(self.name)
            pytest.skip(
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, was already exceeded"
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.session.increment_num_passed(self.name)
        else:
            exceptions_to_ignore = self.session.get_exceptions_to_ignore(self.name)
            if not any(exc_type is x for x in exceptions_to_ignore):
                self.session.increment_num_failures(self.name)
        if self.session.has_exceeded_max_failures(self.name):
            pytest.fail(
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, was exceeded"
            )
        return True


def wrap_pytest_function(pyfuncitem: pytest.Function):
    testfunction = pyfuncitem.obj
    tolerate_obj = TolerateErrorFixture(pyfuncitem)

    @functools.wraps(testfunction)
    def wrapper(*args, **kwargs):
        with tolerate_obj:
            testfunction(*args, **kwargs)

    pyfuncitem.obj = wrapper
