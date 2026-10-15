# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Running the examples
====================

The documentation is built by running the examples, and the build is the
only thing that runs them. These tests run them too.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

#: Root of the source tree.
ROOT = pathlib.Path(__file__).parents[2]

#: The examples the documentation is built from.
EXAMPLES = sorted((ROOT / 'examples').glob('ex_*.py'))


def test_the_examples_are_found() -> None:
    """The source tree holds the examples."""
    assert EXAMPLES


@pytest.mark.parametrize('script', EXAMPLES, ids=lambda item: item.name)
def test_the_example_runs(script: pathlib.Path) -> None:
    """An example runs to its end."""
    subprocess.run([sys.executable, str(script)],
                   check=True,
                   env=dict(os.environ, PYTHONPATH=str(ROOT)))
