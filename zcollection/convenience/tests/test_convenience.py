# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the convenience functions
=================================
"""
import copy
import json
import logging

import numpy
import pytest

from ... import convenience, dataset, meta, variable, view
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.data import create_test_collection
from ...tests.fs import local_fs, s3, s3_base, s3_fs  # noqa: F401

# pylint: disable=unused-import


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_update_deprecated_view(fs, request, caplog):
    """Test the deprecated view update functionality."""
    caplog.set_level(logging.WARNING)
    tested_fs = request.getfixturevalue(fs)
    create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = convenience.create_view(path=str(tested_fs.view),
                                    view_ref=view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs)

    var = meta.Variable(name='var3',
                        dtype=numpy.float64,
                        dimensions=('num_lines', 'num_pixels'))

    zview.add_variable(var)

    def to_zero(zds: dataset.Dataset, varname):
        """Update function used to set a variable to 0."""
        return {varname: zds.variables['var1'].values * 0}

    zview.update(to_zero, var.name)  # type: ignore

    zdata_ref = zview.load(selected_variables=['var1', var.name],
                           delayed=False,
                           distributed=False)

    # Manually editing the collection's configuration to make
    # it deprecated
    #: pylint: disable=protected-access
    conf_zview = zview._config(zview.base_dir)

    with tested_fs.open(conf_zview) as stream:
        cdata_zview = json.load(stream)

    cdata_zview_deprecated = copy.deepcopy(cdata_zview)

    # Removing version number
    del cdata_zview_deprecated['version']

    # Removing size and chunk information from
    # dimensions
    cdata_zview_deprecated['metadata']['dimensions'] = [
        d[0] for d in cdata_zview_deprecated['metadata']['dimensions']
    ]

    with tested_fs.open(conf_zview, mode='w') as stream:
        json.dump(cdata_zview_deprecated, stream, indent=4)

    # Invalid view path
    with pytest.raises(ValueError, match='View not found'):
        convenience.update_deprecated_view(
            path=str(tested_fs.view) + 'xyz',
            filesystem=tested_fs.fs,
        )

    # The view cannot be opened without an update
    with pytest.raises(ValueError, match='needs to be updated'):
        convenience.open_view(path=str(tested_fs.view),
                              filesystem=tested_fs.fs)

    caplog.clear()
    convenience.update_deprecated_view(path=str(tested_fs.view),
                                       filesystem=tested_fs.fs)

    assert 'Updating view' in caplog.text
    assert 'Copying old configuration to' in caplog.text
    assert 'Writing new configuration' in caplog.text

    conf_zview_old = f'{conf_zview}.bak'

    assert tested_fs.exists(conf_zview_old)

    # Configuration is back to what it was
    with tested_fs.open(conf_zview_old) as stream:
        assert json.load(stream) == cdata_zview_deprecated

    # Configuration is back to what it was
    with tested_fs.open(conf_zview) as stream:
        assert json.load(stream) == cdata_zview

    caplog.clear()
    convenience.update_deprecated_view(path=str(tested_fs.view),
                                       filesystem=tested_fs.fs)
    assert 'Updating view' in caplog.text
    assert 'already updated' in caplog.text

    zview = convenience.open_view(path=str(tested_fs.view),
                                  filesystem=tested_fs.fs)
    zdata = zview.load(selected_variables=['var1', var.name],
                       delayed=False,
                       distributed=False)

    for v in ['var1', var.name]:
        assert numpy.allclose(zdata[v].values, zdata_ref[v].values, rtol=0)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_update_deprecated_collection(fs, request, caplog):
    """Test the deprecated collection update functionality."""
    caplog.set_level(logging.WARNING)
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)

    # Adding a new dimension and an immutable variable
    new_dim = meta.Dimension(name='x', value=2, chunks=1)
    zcol.add_dimension(dimension=new_dim)

    known_dimensions, _ = zcol.dimensions_properties()
    dim = 'num_pixels'
    new_data = numpy.arange(known_dimensions[dim] * new_dim.value,
                            dtype='int16').reshape(known_dimensions[dim],
                                                   new_dim.value)

    new_x = variable.Array(name='var_immutable_x',
                           data=new_data,
                           dimensions=(dim, new_dim.name))

    zcol.add_variable(variable=new_x, distributed=False)
    zcol.update_immutable(name=new_x.name, data=new_x.values)

    zdata_ref = zcol.load(delayed=False, distributed=False)

    # Manually editing the collection's configuration to make
    # it deprecated
    base_dir = zcol.partition_properties.dir
    #: pylint: disable=protected-access
    conf_zcol = zcol._config(base_dir)

    with tested_fs.open(conf_zcol) as stream:
        cdata_zcol = json.load(stream)

    cdata_zcol_deprecated = copy.deepcopy(cdata_zcol)

    # Removing version number
    del cdata_zcol_deprecated['version']

    # Removing size and chunk information from
    # dimensions
    cdata_zcol_deprecated['dataset']['dimensions'] = [
        d[0] for d in cdata_zcol['dataset']['dimensions']
    ]
    cdata_zcol_deprecated['dataset']['chunks'] = [
        [d[0], d[2]] for d in cdata_zcol['dataset']['dimensions']
    ]

    with tested_fs.open(conf_zcol, mode='w') as stream:
        json.dump(cdata_zcol_deprecated, stream, indent=4)
    # The collection cannot be opened in write mode without
    # an update
    with pytest.raises(ValueError, match='needs to be updated'):
        convenience.open_collection(path=str(tested_fs.collection),
                                    filesystem=tested_fs.fs,
                                    mode='w')

    with pytest.warns(UserWarning, match='needs to be updated'):
        zcol = convenience.open_collection(path=str(tested_fs.collection),
                                           filesystem=tested_fs.fs)

    zdata = zcol.load(delayed=False, distributed=False)

    for v in zdata_ref.variables:
        if v == 'time':
            assert numpy.array_equal(zdata[v].values, zdata_ref[v].values)
        else:
            assert numpy.allclose(zdata[v].values,
                                  zdata_ref[v].values,
                                  equal_nan=True,
                                  rtol=0)

    # Invalid collection path
    with pytest.raises(ValueError, match='collection not found'):
        convenience.update_deprecated_collection(
            path=str(tested_fs.collection) + 'xyz',
            filesystem=tested_fs.fs,
        )
    caplog.clear()
    convenience.update_deprecated_collection(path=str(tested_fs.collection),
                                             filesystem=tested_fs.fs)

    assert 'Updating collection' in caplog.text
    assert 'Copying old configuration to' in caplog.text
    assert 'Writing new configuration' in caplog.text

    conf_zcol_old = f'{conf_zcol}.bak'

    assert tested_fs.exists(conf_zcol_old)

    # Configuration is back to what it was
    with tested_fs.open(conf_zcol_old) as stream:
        assert json.load(stream) == cdata_zcol_deprecated

    # Configuration is back to what it was
    with tested_fs.open(conf_zcol) as stream:
        assert json.load(stream) == cdata_zcol

    caplog.clear()
    convenience.update_deprecated_collection(path=str(tested_fs.collection),
                                             filesystem=tested_fs.fs)
    assert 'Updating collection' in caplog.text
    assert 'already updated' in caplog.text

    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       filesystem=tested_fs.fs,
                                       mode='w')

    zdata = zcol.load(delayed=False, distributed=False)

    for v in zdata_ref.variables:
        if v == 'time':
            assert numpy.array_equal(zdata[v].values, zdata_ref[v].values)
        else:
            assert numpy.allclose(zdata[v].values,
                                  zdata_ref[v].values,
                                  equal_nan=True,
                                  rtol=0)
