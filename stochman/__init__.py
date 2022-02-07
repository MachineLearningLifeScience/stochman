#!/usr/bin/env python
"""Root package info."""

import logging as python_logging
import os
import time

# fmt: off
__name__ = "stochman"
_this_year = time.strftime("%Y")
__version__ = "0.1.1"
__author__ = "Nicki Skafte Detlefsen et al."
__author_email__ = "nsde@dtu.dk"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{_this_year}, {__author__}."
__homepage__ = "https://github.com/CenterBioML/stochman"
__docs__ = (
    "StochMan is a collection of elementary algorithms for"
    " computations on random manifolds"
)
# fmt: on

_logger = python_logging.getLogger("stochman")
_logger.addHandler(python_logging.StreamHandler())
_logger.setLevel(python_logging.INFO)

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build process
    _ = None if __STOCHMAN_SETUP__ else None
except NameError:
    __STOCHMAN_SETUP__: bool = False

if __STOCHMAN_SETUP__:  # pragma: no cover
    import sys

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")
    # We are not importing the rest of the package during the build process, as it may not be compiled yet
else:
    # import modules
    from . import curves, discretized_manifold, geodesic, manifold, nnj

    # import classes for direct access
    from .curves import CubicSpline, DiscreteCurve
