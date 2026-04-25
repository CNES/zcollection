# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Entry point for building, distributing and installing this module.

Uses distutils/setuptools.
"""

import pathlib

import setuptools
import setuptools.command.sdist

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()


class SDist(setuptools.command.sdist.sdist):
    """Copy the pytest configuration file into the package.

    Copies the pytest configuration file into the package.
    """

    user_options = setuptools.command.sdist.sdist.user_options

    def run(self):
        """Carry out the action."""
        source = WORKING_DIRECTORY.joinpath("conftest.py")
        target = WORKING_DIRECTORY.joinpath("zcollection", "conftest.py")
        source.rename(target)
        try:
            super().run()
        finally:
            target.rename(source)


setuptools.setup(cmdclass={"sdist": SDist})
