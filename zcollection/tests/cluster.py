# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import logging

import dask.config
import dask.distributed
import pytest


@pytest.fixture()
def dask_client(pytestconfig, tmpdir, request):
    """Launch a Dask LocalCluster with a configurable number of workers."""
    try:
        n_workers = int(pytestconfig.getoption("n_workers"))
    except TypeError:
        n_workers = None

    try:
        threads_per_worker = int(pytestconfig.getoption("threads_per_worker"))
    except TypeError:
        threads_per_worker = None

    # We don't want to create temporary files in the source tree.
    dask.config.set(temporary_directory=str(tmpdir))
    cluster = dask.distributed.LocalCluster(
        protocol="tcp://",
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=False)

    # Make sure we can connect to the cluster.
    client = dask.distributed.Client(cluster)
    client.wait_for_workers(1)

    def teardown():
        """Teardown the cluster."""
        client.close()
        cluster.close()

    request.addfinalizer(teardown)

    return client
