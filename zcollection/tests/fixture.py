# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixtures for the tests.
=======================
"""
from __future__ import annotations

from typing import Literal

import pytest

from .. import collection, convenience, meta, partitioning, view


@pytest.fixture
def dask_arrays() -> Literal[True]:
    """Load the data in Dask arrays."""
    return True


@pytest.fixture
def numpy_arrays() -> Literal[False]:
    """Load the data in NumPy arrays."""
    return False


@pytest.fixture(params=['collection', 'view'])
def empty_target(request, local_fs) -> collection.Collection | view.View:
    """Create an empty collection or a view with a declared output."""
    key = meta.VariableSchema('key', 'int64', dimensions=('row', ))
    schema = meta.DatasetSchema([meta.Dimension('row', -1)], [key])
    source = str(local_fs.collection)
    zcol = collection.Collection('key',
                                 schema,
                                 partitioning.Sequence(('key', )),
                                 source,
                                 filesystem=local_fs.fs)
    if request.param == 'collection':
        return zcol
    zview = convenience.create_view(str(local_fs.view),
                                    view.ViewReference(source, local_fs.fs),
                                    filesystem=local_fs.fs,
                                    distributed=False)
    zview.add_variable(meta.VariableSchema('output',
                                           'int64',
                                           dimensions=('row', )),
                       distributed=False)
    return zview
