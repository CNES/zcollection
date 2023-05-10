# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixtures for testing S3 using the pytest and minio.
===================================================
"""
from typing import Iterator, Literal
import os
import pathlib
import shlex
import subprocess
import time

import botocore.client
import botocore.session
import pytest
import requests
import s3fs

#: Listen port
PORT = 5555
#: Listen address
ENDPOINT: str = f'127.0.0.1:{PORT}'
#: URI for minio
ENDPOINT_URI: str = f'http://{ENDPOINT}'
#: Credential for minio
CREDENTIAL = '25219d58-f6c6-11eb-922c-770d49cd18e4'


def have_minio() -> Literal[True]:
    """Check if minio is available."""
    try:
        subprocess.check_output(['minio', '--version'])
        return True
    except:
        raise ImportError('minio: command not found') from None


have_minio()


def is_minio_up(timeout: float) -> bool:
    """Check if minio server is up."""
    try:
        response = requests.get(ENDPOINT_URI, timeout=timeout)
        if response.status_code == 403:
            return True
    except:  # pylint: disable=bare-except
        pass
    return False


def wait_for_minio_to_start(timeout: float) -> None:
    """Wait for the minio server to start."""
    while timeout > 0:
        try:
            response = requests.get(ENDPOINT_URI, timeout=1)
            if response.status_code == 403:
                return
        except:  # pylint: disable=bare-except
            pass
        timeout -= 0.1
        time.sleep(0.1)
    raise RuntimeError("minio server didn't start")


@pytest.fixture()
def s3_base(tmpdir, pytestconfig) -> Iterator[None]:
    """Launch minio server."""
    if pytestconfig.getoption('s3') is False:
        pytest.skip('S3 disabled')
    if is_minio_up(timeout=1):
        raise RuntimeError('minio server already up')
    os.environ['MINIO_CACHE_AFTER'] = '1'
    os.environ['MINIO_CACHE'] = 'on'
    os.environ['MINIO_ROOT_PASSWORD'] = CREDENTIAL
    os.environ['MINIO_ROOT_USER'] = CREDENTIAL
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        shlex.split(f'minio server --quiet --address {ENDPOINT} '
                    f"--console-address :{PORT+1} '{tmpdir!s}'"))

    try:
        wait_for_minio_to_start(timeout=30)
        yield
    finally:
        process.terminate()
        process.wait()
    # pylint: enable=consider-using-with


def make_bucket(name) -> None:
    """Create a bucket."""
    session: botocore.session.Session = botocore.session.get_session()
    client = session.create_client(
        's3',
        aws_access_key_id=CREDENTIAL,
        aws_secret_access_key=CREDENTIAL,
        endpoint_url=ENDPOINT_URI,
        region_name='us-east-1',
        config=botocore.client.Config(signature_version='s3v4'))
    client.create_bucket(Bucket=name, ACL='public-read')


# pylint: disable=redefined-outer-name, unused-argument # pytest fixture
@pytest.fixture()
def s3(s3_base) -> Iterator[s3fs.core.S3FileSystem]:
    """Create a S3 file system instance."""
    s3fs.core.S3FileSystem.clear_instance_cache()
    fs = s3fs.core.S3FileSystem(anon=False,
                                key=CREDENTIAL,
                                secret=CREDENTIAL,
                                client_kwargs={'endpoint_url': ENDPOINT_URI})
    fs.invalidate_cache()
    yield fs
    # pylint: enable=redefined-outer-name, unused-argument


class S3Path(type(pathlib.Path())):  # type: ignore[misc]
    """Handle S3 path on multiple platforms."""

    def __str__(self) -> str:
        return super().__str__().replace('\\', '/')


class S3:
    """S3 filesystem."""
    #: Bucket ID
    ID = 0

    # pylint: disable=redefined-outer-name # pytest fixture
    def __init__(self, s3: s3fs.core.S3FileSystem) -> None:
        name: str = f'bucket{S3.ID}'
        S3.ID += 1
        make_bucket(name)
        self.collection: S3Path = S3Path(name).joinpath('collection')
        self.view: S3Path = S3Path(name).joinpath('view')
        self.fs: s3fs.core.S3FileSystem = s3

    # pylint: enable=redefined-outer-name
