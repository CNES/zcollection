# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Build, distribute, and install the package using setuptools."""
import pathlib

import setuptools
import setuptools.command.sdist

# Directory containing setup.py.
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()


class SDist(setuptools.command.sdist.sdist):
    """Build a source distribution with the pytest configuration in the package.

    Move ``conftest.py`` into the package before building, then restore it
    to its original location.
    """
    user_options = setuptools.command.sdist.sdist.user_options

    def run(self):
        """Run the source distribution build."""
        source = WORKING_DIRECTORY.joinpath('conftest.py')
        target = WORKING_DIRECTORY.joinpath('zcollection', 'conftest.py')
        source.rename(target)
        try:
            super().run()
        finally:
            target.rename(source)


setuptools.setup(
    cmdclass={'sdist': SDist},  # type: ignore[arg-type]
)
