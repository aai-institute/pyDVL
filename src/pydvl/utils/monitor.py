"""
This module implements a simple memory monitoring utility for the whole application.

With [start_memory_monitoring()][pydvl.utils.monitor.start_memory_monitoring] one can
monitor global memory usage, including the memory of child processes. The monitoring
runs in a separate thread and keeps track of the *maximum** memory usage observed.

Monitoring stops automatically when the process exits or receives common termination
signals (SIGINT, SIGTERM, SIGHUP). It can also be stopped manually by calling
[end_memory_monitoring()][pydvl.utils.monitor.end_memory_monitoring].

When monitoring stops, the maximum memory usage is both logged and returned (in bytes).

!!! note
    This is intended to report peak memory usage for the whole application, including
    child processes. It is not intended to be used for profiling memory usage of
    individual functions or modules. Given that there exist numerous profiling tools,
    it probably doesn't make sense to extend this module further.
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
import time
from collections import defaultdict
from itertools import chain

import psutil

__all__ = [
    "end_memory_monitoring",
    "log_memory_usage_report",
    "start_memory_monitoring",
]

logger = logging.getLogger(__name__)

__state_lock = threading.Lock()
__memory_usage: defaultdict[int, int] = defaultdict(int)  # pid -> bytes
__peak_memory_usage = 0  # (in bytes)
__monitoring_enabled = threading.Event()
__memory_monitor_thread: threading.Thread | None = None


def _memory_monitor_thread() -> threading.Thread | None:
    """Returns the memory monitor thread. Can be None if the monitor was never started.
    This is only useful for testing purposes."""
    return __memory_monitor_thread


def start_memory_monitoring(auto_stop: bool = True):
    """Starts a memory monitoring thread.

    The monitor runs in a separate thread and keeps track of maximum memory usage
    observed during the monitoring period.

    The monitoring stops by calling
    [end_memory_monitoring()][pydvl.utils.monitor.end_memory_monitoring] or, if
    `auto_stop` is `True` when the process is terminated or exits.

    Args:
        auto_stop: If True, the monitoring will stop when the process exits
            normally or receives common termination signals (SIGINT, SIGTERM, SIGHUP).

    """
    global __memory_usage
    global __memory_monitor_thread
    global __peak_memory_usage

    if __monitoring_enabled.is_set():
        logger.warning("Memory monitoring is already running.")
        return

    with __state_lock:
        __memory_usage.clear()
        __peak_memory_usage = 0

    __monitoring_enabled.set()
    __memory_monitor_thread = threading.Thread(
        target=memory_monitor_run, args=(psutil.Process().pid,)
    )
    __memory_monitor_thread.start()

    if not auto_stop:
        return

    atexit.register(end_memory_monitoring)

    # Register signal handlers for common termination signals, re-raising the original
    # signal to terminate as expected

    def signal_handler(signum, frame):
        end_memory_monitoring()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    # SIGHUP might not be available on all platforms (e.g., Windows)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)  # Terminal closed


def memory_monitor_run(pid: int, interval: float = 0.1):
    """Monitors the memory usage of the process and its children.

    This function runs in a separate thread and updates the global variable
    `__max_memory_usage` with the maximum memory usage observed during the monitoring
    period.

    The monitoring stops when the __monitoring_enabled event is cleared, which can be
    achieved either by calling
    [end_memory_monitoring()][pydvl.utils.monitor.end_memory_monitoring], or when the
    process is terminated or exits.
    """
    global __memory_usage
    global __peak_memory_usage

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.error(f"Process {pid} not found. Monitoring cannot start.")
        return

    while __monitoring_enabled.is_set():
        total_mem = 0
        try:
            for p in chain([proc], proc.children(recursive=True)):
                try:
                    pid = p.pid
                    rss = p.memory_info().rss
                    total_mem += rss
                    with __state_lock:
                        __memory_usage[pid] = max(__memory_usage[pid], rss)
                except psutil.NoSuchProcess:
                    continue
        except psutil.NoSuchProcess:  # Catch invalid proc / proc.children
            break

        with __state_lock:
            __peak_memory_usage = max(__peak_memory_usage, total_mem)

        time.sleep(interval)


def end_memory_monitoring(log_level=logging.DEBUG) -> tuple[int, dict[int, int]]:
    """Ends the memory monitoring thread and logs the maximum memory usage.

    Args:
        log_level: The logging level to use.

    Returns:
        A tuple with the maximum memory usage observed globally, and for each pid
            separately as a dict. The dict will be empty if monitoring is disabled.
    """
    global __memory_usage
    global __peak_memory_usage

    if not __monitoring_enabled.is_set():
        return 0, {}

    __monitoring_enabled.clear()
    assert __memory_monitor_thread is not None
    __memory_monitor_thread.join()

    with __state_lock:
        peak_mem = __peak_memory_usage
        mem_usage = __memory_usage.copy()
        __memory_usage.clear()
        __peak_memory_usage = 0

    log_memory_usage_report(peak_mem, mem_usage, log_level)
    return peak_mem, mem_usage


def log_memory_usage_report(
    peak_mem: int, mem_usage: dict[int, int], log_level=logging.DEBUG
):
    """
    Generates a nicely tabulated memory usage report and logs it.

    Args:
        peak_mem: The maximum memory usage observed during the monitoring period.
        mem_usage: A dictionary mapping process IDs (pid) to memory usage in bytes.
        log_level: The log level used for logging the report.
    """
    if not mem_usage:
        logger.log(log_level, "No memory usage data available.")
        return

    headers = ("PID", "Memory (Bytes)", "Memory (MB)")
    col_widths = (10, 20, 15)

    header_line = (
        f"{headers[0]:>{col_widths[0]}} "
        f"{headers[1]:>{col_widths[1]}} "
        f"{headers[2]:>{col_widths[2]}}"
    )
    separator = "-" * (sum(col_widths) + 2)

    summary = (
        f"Memory monitor: {len(mem_usage)} processes monitored. "
        f"Peak memory usage: {peak_mem / (2**20):.2f} MB"
    )

    lines = [header_line, separator, summary]

    for pid, bytes_used in sorted(
        mem_usage.items(), key=lambda item: item[1], reverse=True
    ):
        mb_used = bytes_used / (1024 * 1024)
        line = (
            f"{pid:>{col_widths[0]}} "
            f"{bytes_used:>{col_widths[1]},} "
            f"{mb_used:>{col_widths[2]}.2f}"
        )
        lines.append(line)

    lines.append(separator)

    logger.log(log_level, "\n".join(lines))
