# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
File system test fixtures
=========================
"""
from typing import Any
from collections.abc import Iterator
import pathlib
import tempfile

import fsspec
import fsspec.implementations.memory
import pytest

try:

    from .s3 import S3, s3, s3_base

    S3_IMPORT_EXCEPTION = None
except ImportError as err:
    S3_IMPORT_EXCEPTION = str(err)


def tempdir(tmp_path) -> pathlib.Path:
    """Create a temporary directory."""
    return pathlib.Path(tempfile.mkdtemp(dir=str(tmp_path)))


class Local:
    """Local file system."""

    def __init__(self, tmp_path, protocol) -> None:
        #: The file system.
        self.fs: fsspec.AbstractFileSystem = fsspec.filesystem(protocol)
        #: The root directory.
        self.root = tempdir(pathlib.Path(tmp_path))
        #: The collection directory.
        self.collection: pathlib.Path = self.root.joinpath('collection')
        #: The view directory.
        self.view: pathlib.Path = self.root.joinpath('view')

    def __getattr__(self, name) -> Any:
        return getattr(self.fs, name)


@pytest.fixture
def local_fs(tmp_path, pytestconfig) -> Iterator[Local]:
    """Provide a local file system."""
    protocol: str = 'memory' if pytestconfig.getoption('memory') else 'file'
    instance = Local(tmp_path, protocol)
    yield instance
    try:
        # Remove test data on the memory file system to release memory.
        if isinstance(instance.fs,
                      fsspec.implementations.memory.MemoryFileSystem):
            instance.fs.rm(str(instance.root), recursive=True)
    except FileNotFoundError:
        pass


if S3_IMPORT_EXCEPTION is None:

    @pytest.fixture
    def s3_fs(s3, s3_base) -> S3:  # type: ignore[arg-type]
        """S3 file system."""
        return S3(s3, s3_base)
else:

    @pytest.fixture
    def s3() -> None:
        """S3 file system."""

    @pytest.fixture
    def s3_base() -> None:
        """S3 file system."""

    @pytest.fixture
    def s3_fs(pytestconfig) -> None:
        """S3 file system."""
        if pytestconfig.getoption('s3'):
            pytest.fail(f'Unable to test S3: {S3_IMPORT_EXCEPTION}')
        else:
            pytest.skip('S3 is disabled')
