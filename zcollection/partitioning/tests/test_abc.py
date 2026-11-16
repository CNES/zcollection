# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Base partitioning tests
======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import fsspec
import pytest

from .. import Date, Sequence

if TYPE_CHECKING:
    from ..abc import Partitioning


@pytest.mark.parametrize(
    ('partitioning', 'valid_dirs', 'stray_root', 'stray_intermediate'), [
        pytest.param(Date(('time', ), 'Y'), ['year=2020', 'year=2021'],
                     'incomplete_upload',
                     None,
                     id='year'),
        pytest.param(Date(
            ('time', ),
            'D'), ['year=2023/month=01/day=01', 'year=2023/month=01/day=02'],
                     'incomplete_upload',
                     'year=2023/month=01/.tmp_incomplete',
                     id='day'),
        pytest.param(
            Sequence(('cycle_number', 'pass_number')),
            ['cycle_number=1/pass_number=10', 'cycle_number=1/pass_number=20'],
            'garbage_dir',
            'cycle_number=1/.tmp_write',
            id='sequence'),
    ])
def test_list_partitions_tolerates_stray_directories(
    partitioning: Partitioning,
    valid_dirs: list[str],
    stray_root: str,
    stray_intermediate: str | None,
) -> None:
    """Ignore stray directories at the root and within partition levels."""
    fs = fsspec.filesystem('memory')
    root = f'/partitioning-{uuid.uuid4().hex}'
    fs.mkdir(root)
    try:
        expected = [f'{root}/{item}' for item in valid_dirs]
        for partition in expected:
            fs.mkdirs(partition)

        fs.mkdirs(f'{root}/{stray_root}')
        if stray_intermediate is not None:
            fs.mkdirs(f'{root}/{stray_intermediate}')

        assert list(partitioning.list_partitions(fs, root)) == expected
    finally:
        fs.rm(root, recursive=True)
