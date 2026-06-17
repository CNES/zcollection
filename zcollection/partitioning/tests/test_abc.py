# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test base partitioning robustness.

===================================
"""

from __future__ import annotations

import random
import string

import fsspec
import pytest

from .. import Date, Sequence


def _random_root() -> str:
    return '/' + ''.join(random.choices(string.ascii_letters, k=10))


@pytest.mark.parametrize(
    'partitioning,valid_dirs,stray_root,stray_intermediate', [
        (
            Date(('time', ), 'Y'),
            ['year=2020', 'year=2021'],
            'incomplete_upload',
            None,
        ),
        (
            Date(('time', ), 'D'),
            ['year=2023/month=01/day=01', 'year=2023/month=01/day=02'],
            'incomplete_upload',
            'year=2023/month=01/.tmp_incomplete',
        ),
        (
            Sequence(('cycle_number', 'pass_number')),
            ['cycle_number=1/pass_number=10', 'cycle_number=1/pass_number=20'],
            'garbage_dir',
            'cycle_number=1/.tmp_write',
        ),
    ])
def test_list_partitions_tolerates_stray_directories(
    partitioning,
    valid_dirs,
    stray_root,
    stray_intermediate,
) -> None:
    """Stray/incomplete directories must not crash list_partitions."""
    fs = fsspec.filesystem('memory')
    root = _random_root()
    fs.mkdir(root)

    expected = [f'{root}/{d}' for d in valid_dirs]
    for p in expected:
        fs.mkdirs(p)

    fs.mkdirs(f'{root}/{stray_root}')
    if stray_intermediate is not None:
        fs.mkdirs(f'{root}/{stray_intermediate}')

    assert list(partitioning.list_partitions(fs, root)) == expected
