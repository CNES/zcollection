# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixture for testing the file system.
====================================
"""
import pathlib
import uuid

import fsspec
import pytest

try:
    # pylint: disable=unused-import # Need to import for fixtures
    from .s3 import S3, s3, s3_base  # type: ignore

    # pylint: disable=unused-import
    S3_ENABLED = None
except ImportError as err:
    S3_ENABLED = str(err)


class Local:
    """Local filesystem"""

    def __init__(self, tmpdir) -> None:
        self.rool = pathlib.Path(tmpdir)
        self.collection = pathlib.Path(tmpdir).joinpath("collection")
        self.view = pathlib.Path(tmpdir).joinpath("view")
        self.fs = fsspec.filesystem("file")


class Memory:
    """Memory filesystem"""

    def __init__(self) -> None:
        self.fs = fsspec.filesystem("memory")
        self.root = self.fs.sep.join(("", str(uuid.uuid4())))
        self.collection = self.fs.sep.join((self.root, "collection"))
        self.view = self.fs.sep.join((self.root, "view"))

    def __getattr__(self, name):
        return getattr(self.fs, name)


@pytest.fixture
def local_fs(tmpdir):
    """Local filesystem"""
    return Local(tmpdir)


@pytest.fixture
def memory_fs():
    """Memory filesystem"""
    instance = Memory()
    instance.fs.mkdir(instance.root)
    yield instance
    try:
        instance.fs.rm(instance.root, recursive=True)
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
