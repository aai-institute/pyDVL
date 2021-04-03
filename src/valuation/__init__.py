__version__ = "0.1.0-dev1"

_logger = None


def set_logger(logger=None):
    global _logger
    if logger is not None:
        _logger = logger
    else:
        import logging
        _logger = logging.getLogger()


set_logger()
