# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
==========
Test setup
==========
"""


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption("--minio", action="store_true", default=False)
    parser.addoption("--threads_per_worker", action="store", default=None)
    parser.addoption("--n_workers", action="store", default=None)
