# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import contextlib
import logging
import weakref

import dask.config
import dask.distributed
import pytest


@pytest.fixture()
def dask_cluster(pytestconfig, tmpdir_factory, scope='session'):
    """Launch a Dask LocalCluster with a configurable number of workers."""
    try:
        n_workers = int(pytestconfig.getoption('n_workers'))
    except TypeError:
        n_workers = None

    try:
        threads_per_worker = int(pytestconfig.getoption('threads_per_worker'))
    except TypeError:
        threads_per_worker = None

    try:
        processes = int(pytestconfig.getoption('processes')) == 1
    except TypeError:
        processes = False

    tmpdir = tmpdir_factory.getbasetemp()
    scheduler_file = tmpdir / 'scheduler.json'
    if scheduler_file.exists():
        return str(scheduler_file)

    # Use the root path of the test session for the dask worker space
    dask_worker = tmpdir / 'dask_worker_space'
    dask.config.set(temporary_directory=str(dask_worker))

    logging.info('Dask local cluster starting')
    cluster = dask.distributed.LocalCluster(
        protocol='tcp://',
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes)

    def teardown():
        """Stop the cluster and remove the scheduler file."""
        cluster.close()
        if scheduler_file.exists():
            scheduler_file.remove()

    weakref.finalize(cluster, teardown)

    # Make sure we can connect to the cluster.
    with dask.distributed.Client(cluster) as client:
        client.write_scheduler_file(scheduler_file)
        client.wait_for_workers(1)

    logging.info('Dask local cluster started')
    return str(scheduler_file)


@contextlib.contextmanager
def _scheduler_file(dask_cluster):
    """Get the scheduler used to connect to the cluster."""
    yield dask_cluster


@pytest.fixture()
def dask_client(dask_cluster):
    """Connect a Dask client to the cluster."""
    with _scheduler_file(dask_cluster) as scheduler_file:
        with dask.distributed.Client(scheduler_file=scheduler_file) as client:
            yield client
