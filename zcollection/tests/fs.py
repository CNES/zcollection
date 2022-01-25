# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixture for testing the file system.
====================================
"""
import pathlib

import fsspec
import fsspec.implementations.memory
import pytest

try:
    # pylint: disable=unused-import # Need to import for fixtures
    from .s3 import S3, s3, s3_base  # type: ignore

    # pylint: disable=unused-import
    S3_ENABLED = None
except ImportError as err:
    S3_ENABLED = str(err)


class Local:
    """Local files system"""

    def __init__(self, tmpdir, protocol) -> None:
        self.fs = fsspec.filesystem(protocol)
        self.root = pathlib.Path(tmpdir)
        self.collection = self.root.joinpath("collection")
        self.view = self.root.joinpath("view")

    def __getattr__(self, name):
        return getattr(self.fs, name)


@pytest.fixture
def local_fs(tmpdir, pytestconfig):
    """Local filesystem"""
    protocol = "memory" if pytestconfig.getoption("memory") else "file"
    instance = Local(tmpdir, protocol)
    yield instance
    try:
        # For the memory protocol we delete the written data to free the
        # memory.
        if isinstance(instance.fs,
                      fsspec.implementations.memory.MemoryFileSystem):
            instance.fs.rm(str(instance.root), recursive=True)
    except FileNotFoundError:
        pass


# pylint: disable=redefined-outer-name,function-redefined
if S3_ENABLED is None:

    @pytest.fixture
    def s3_fs(s3):  # type: ignore (enabled only if S3 is available)
        """S3 filesystem"""
        return S3(s3)
else:

    @pytest.fixture
    def s3():
        """S3 filesystem"""
        ...

    @pytest.fixture
    def s3_base():
        """S3 filesystem"""
        ...

    @pytest.fixture
    def s3_fs():
        """S3 filesystem"""
        assert S3_ENABLED is not None
        pytest.skip("Prerequisites are missing to test S3: " + S3_ENABLED)


# pylint: enable=redefined-outer-name,function-redefined
