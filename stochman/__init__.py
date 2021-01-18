# Copyright The GeoML Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Root package info."""

import logging as python_logging
import os
import time

_this_year = time.strftime("%Y")
__version__ = "0.1.1"
__author__ = "Nicki Skafte Detlefsen et al."
__author_email__ = "nsde@dtu.dk"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{_this_year}, {__author__}."
__homepage__ = "https://github.com/CenterBioML/stochman"

__docs__ = "StochMan is a collection of elementary algorithms for computations on random manifolds"

_logger = python_logging.getLogger("stochman")
_logger.addHandler(python_logging.StreamHandler())
_logger.setLevel(python_logging.INFO)

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

__all__ = []
