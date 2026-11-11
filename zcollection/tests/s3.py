# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixtures for testing S3 using the pytest and minio.
===================================================
"""
from typing import Literal
from collections.abc import Iterator
import os
import pathlib
import socket
import subprocess
import time

import botocore.client
import botocore.session
from fsspec.asyn import sync
import pytest
import requests
import s3fs

#: Credential for minio
CREDENTIAL = '25219d58-f6c6-11eb-922c-770d49cd18e4'


def have_minio() -> Literal[True]:
    """Check if minio is available."""
    try:
        subprocess.check_output(['minio', '--version'])
        return True
    except:  # noqa: E722
        # Ignore exceptions, we are checking if the command is available
        raise ImportError('minio: command not found') from None


have_minio()


def is_minio_up(endpoint_uri: str, timeout: float) -> bool:
    """Check if minio server is up."""
    try:
        response = requests.get(endpoint_uri, timeout=timeout)
        if response.status_code == 403:
            return True
    except:  # noqa: E722
        # Ignore exceptions, we are checking if the server is up
        pass
    return False


def wait_for_minio_to_start(endpoint_uri: str, timeout: float) -> None:
    """Wait for the minio server to start."""
    while timeout > 0:
        try:
            response = requests.get(endpoint_uri, timeout=1)
            if response.status_code == 403:
                return
        except:  # noqa: E722
            # Ignore exceptions, we are waiting for the server to start
            pass
        timeout -= 0.1
        time.sleep(0.1)
    raise RuntimeError("minio server didn't start")


def unused_port() -> int:
    """Return a port selected by the OS after releasing its socket."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='session')
def s3_base(tmp_path_factory, pytestconfig) -> Iterator[str]:
    """Launch minio server."""
    if pytestconfig.getoption('s3') is False:
        pytest.skip('S3 disabled')
    port = unused_port()
    console_port = unused_port()
    while console_port == port:
        console_port = unused_port()
    endpoint = f'127.0.0.1:{port}'
    endpoint_uri = f'http://{endpoint}'
    if is_minio_up(endpoint_uri, timeout=1):
        raise RuntimeError('minio server already up')
    data_dir = tmp_path_factory.mktemp('minio')
    os.environ['MINIO_CACHE_AFTER'] = '1'
    os.environ['MINIO_CACHE'] = 'on'
    os.environ['MINIO_ROOT_PASSWORD'] = CREDENTIAL
    os.environ['MINIO_ROOT_USER'] = CREDENTIAL

    process = subprocess.Popen([
        'minio', 'server', '--quiet', '--address', endpoint,
        '--console-address', f'127.0.0.1:{console_port}',
        str(data_dir)
    ])

    try:
        wait_for_minio_to_start(endpoint_uri, timeout=30)
        yield endpoint_uri
    finally:
        process.terminate()
        process.wait()


def make_bucket(name, endpoint_uri: str) -> None:
    """Create a bucket."""
    session: botocore.session.Session = botocore.session.get_session()
    client = session.create_client(
        's3',
        aws_access_key_id=CREDENTIAL,
        aws_secret_access_key=CREDENTIAL,
        endpoint_url=endpoint_uri,
        region_name='us-east-1',
        config=botocore.client.Config(signature_version='s3v4'))
    try:
        client.create_bucket(Bucket=name, ACL='public-read')
    finally:
        client.close()


@pytest.fixture
def s3(s3_base) -> Iterator[s3fs.core.S3FileSystem]:
    """Create a S3 file system instance."""
    s3fs.core.S3FileSystem.clear_instance_cache()
    fs = s3fs.core.S3FileSystem(anon=False,
                                key=CREDENTIAL,
                                secret=CREDENTIAL,
                                client_kwargs={'endpoint_url': s3_base})
    fs.invalidate_cache()
    try:
        yield fs
    finally:
        creator = getattr(fs, '_s3creator', None)
        try:
            if creator is not None:
                sync(fs.loop, creator.__aexit__, None, None, None, timeout=10)
        finally:
            s3fs.core.S3FileSystem.clear_instance_cache()


class S3Path(type(pathlib.Path())):  # type: ignore[misc]
    """Handle S3 path on multiple platforms."""

    def __str__(self) -> str:
        return super().__str__().replace('\\', '/')


class S3:
    """S3 filesystem."""
    #: Bucket ID
    ID = 0

    def __init__(self, s3: s3fs.core.S3FileSystem, endpoint_uri: str) -> None:
        name: str = f'bucket{S3.ID}'
        S3.ID += 1
        make_bucket(name, endpoint_uri)
        self.collection: S3Path = S3Path(name).joinpath('collection')
        self.view: S3Path = S3Path(name).joinpath('view')
        self.fs: s3fs.core.S3FileSystem = s3
