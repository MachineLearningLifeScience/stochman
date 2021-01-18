#!/usr/bin/env python
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
import os
from io import open

from setuptools import Command, find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

PATH_ROOT = os.path.dirname(__file__)
builtins.__STOCHMAN_SETUP__ = True

import stochman


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


PATH_ROOT = os.path.dirname(__file__)


def load_readme(path_dir=PATH_ROOT):
    with open(os.path.join(path_dir, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name="stochman",
    version=stochman.__version__,
    description=stochman.__docs__,
    long_description=load_readme(PATH_ROOT),
    long_description_content_type="text/markdown",
    author=stochman.__author__,
    author_email=stochman.__author_email__,
    license=stochman.__license__,
    packages=find_packages(exclude=["tests", "tests/*"]),
    python_requires=">=3.8",
    install_requires=["torch>=1.6", "numpy>=1.16.4"],
    download_url="https://github.com/CenterBioML/stochman/archive/0.1.0.zip",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
    ],
)
