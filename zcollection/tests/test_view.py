# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
import numpy
import pytest

from .. import meta, view
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
from .data import create_test_collection
from .fs import local_fs, s3, s3_base, s3_fs

# pylint: enable=unused-import


@pytest.mark.parametrize("arg", ["local_fs", "s3_fs"])
def test_view(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the creation of a view"""
    # import dask.distributed
    # cluster = dask.distributed.LocalCluster(n_workers=1,
    #                                         threads_per_worker=1,
    #                                         processes=False)
    # client = dask.distributed.Client(cluster)
    tested_fs = request.getfixturevalue(arg)

    create_test_collection(tested_fs)
    instance = view.create_view(str(tested_fs.view),
                                view.ViewReference(str(tested_fs.collection),
                                                   tested_fs.fs),
                                filesystem=tested_fs.fs)
    assert isinstance(instance, view.View)
    assert isinstance(str(instance), str)
    var = meta.Variable(
        name="var2",
        dtype=numpy.float64,
        dimensions=("num_lines", "num_pixels"),
        attrs=(meta.Attribute(name="attr", value=1), ),
    )

    with pytest.raises(ValueError):
        instance.add_variable(var)

    var.name = "var3"
    instance.add_variable(var)

    with pytest.raises(ValueError):
        instance.add_variable(var)

    instance = view.open_view(str(tested_fs.view), filesystem=tested_fs.fs)
    ds = instance.load()
    assert ds is not None

    def update(ds_x):
        """Update function used for this test."""
        return ds_x.variables["var1"].values * 0 + 5

    instance.update(update, "var3")

    with pytest.raises(ValueError):
        instance.update(update, "varX")

    with pytest.raises(ValueError):
        instance.update(update, "var2")

    ds = instance.load()
    assert ds is not None
    numpy.all(ds.variables["var3"].values == 5)

    instance.drop_variable("var3")

    with pytest.raises(ValueError):
        view.open_view(str(tested_fs.collection), filesystem=tested_fs.fs)
