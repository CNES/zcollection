# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixtures for testing Dask clusters with pytest.
===============================================
"""
from typing import TYPE_CHECKING
from collections.abc import Iterator
import contextlib
import logging

import dask.config
import dask.distributed
import pytest

if TYPE_CHECKING:
    import py


@pytest.fixture(scope='session')
def dask_cluster(
    pytestconfig,
    tmpdir_factory,
) -> Iterator[str]:
    """Launch a Dask LocalCluster with a configurable number of workers.

    Under xdist each worker uses a separate base directory and
    starts a separate cluster.
    """
    n_workers: int | None
    threads_per_worker: int | None
    processes: bool

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

    tmp_path: py.path.local = tmpdir_factory.getbasetemp()
    scheduler_file: py.path.local = tmp_path / 'scheduler.json'

    # Use the test session's root directory for Dask worker storage.
    dask_worker: py.path.local = tmp_path / 'dask_worker_space'
    dask.config.set(temporary_directory=str(dask_worker))

    logging.info('Dask local cluster starting')
    cluster = dask.distributed.LocalCluster(
        protocol='tcp://',
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        dashboard_address=':0',
    )

    try:
        # Make sure we can connect to the cluster.
        with dask.distributed.Client(cluster) as client:
            client.write_scheduler_file(scheduler_file)
            client.wait_for_workers(1)

        logging.info('Dask local cluster started')
        yield str(scheduler_file)
    finally:
        cluster.close()
        if scheduler_file.exists():
            scheduler_file.remove()


@contextlib.contextmanager
def _scheduler_file(dask_cluster, ) -> Iterator[str]:
    """Yield the scheduler file path used to connect to the cluster."""
    yield dask_cluster


@pytest.fixture
def dask_client(dask_cluster, ) -> Iterator[dask.distributed.Client]:
    """Connect a Dask client to the cluster."""
    with _scheduler_file(dask_cluster) as scheduler_file:
        try:
            with dask.distributed.Client(
                    scheduler_file=scheduler_file) as client:
                yield client
        except RuntimeError:
            # Ignore while the client is being stopped.
            pass
