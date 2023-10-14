"""
# This module is deprecated

!!! warning "Redirects"
    Imports from this module will be redirected to
    [pydvl.parallel][pydvl.parallel] only until v0.9.0. Please update your
    imports.
"""
import logging

from ..parallel.backend import *
from ..parallel.config import *
from ..parallel.futures import *
from ..parallel.map_reduce import *

log = logging.getLogger(__name__)

# This string for the benefit of deprecation searches:
# remove_in="0.9.0"
log.warning(
    "Importing parallel tools from pydvl.utils is deprecated. "
    "Please import directly from pydvl.parallel. "
    "Redirected imports will be removed in v0.9.0"
)
