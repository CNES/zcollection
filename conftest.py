# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
==========
Test setup
==========
"""

from zcollection.tests.fs import s3, s3_base, s3_fs  # noqa: F401


def pytest_addoption(parser) -> None:
    """Add command line options to pytest."""
    parser.addoption(
        '--s3',
        action='store_true',
        default=False,
        help='Enable tests on the local S3 server driven by minio. '
        '(default: False)')
    parser.addoption(
        '--memory',
        action='store_true',
        default=False,
        help='Use a file system in memory instead of the local file system. '
        '(default: False)')
    parser.addoption(
        '--threads_per_worker',
        action='store',
        default=None,
        type=int,
        help='Number of threads per Dask worker. (default: chosen by Dask '
        'based on the available CPU cores).')
    parser.addoption(
        '--n_workers',
        action='store',
        default=None,
        type=int,
        help='Number of Dask workers. (default: chosen by Dask based on the '
        'available CPU cores).')
    parser.addoption(
        '--processes',
        action='store_true',
        default=False,
        help='Whether to use processes or threads for Dask. (default: False)')
