# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Support to test with minio
==========================
"""
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
ENDPOINT = f'127.0.0.1:{PORT}'
#: URI for minio
ENDPOINT_URI = f'http://{ENDPOINT}'
#: Credential for minio
CREDENTIAL = '25219d58-f6c6-11eb-922c-770d49cd18e4'


def have_minio():
    """Check if minio is available."""
    try:
        subprocess.check_output(['minio', '--version'])
        return True
    except:
        raise ImportError('minio: command not found') from None


have_minio()


@pytest.fixture()
def s3_base(tmpdir, pytestconfig):
    """Launch minio server."""
    if pytestconfig.getoption('s3') is False:
        pytest.skip('S3 disabled')
    try:
        # should fail since we didn't start server yet
        response = requests.get(ENDPOINT_URI)
    # pylint: disable=bare-except
    except:
        pass
    # pylint: enable=bare-except
    else:
        if response.status_code == 403:
            raise RuntimeError('minio server already up')
    os.environ['MINIO_CACHE_AFTER'] = '1'
    os.environ['MINIO_CACHE'] = 'on'
    os.environ['MINIO_ROOT_PASSWORD'] = CREDENTIAL
    os.environ['MINIO_ROOT_USER'] = CREDENTIAL
    # pylint: disable=consider-using-with
    process = subprocess.Popen(
        shlex.split(f'minio server --quiet --address {ENDPOINT} '
                    f"--console-address :{PORT+1} '{tmpdir!s}'"))

    timeout = 5
    while timeout > 0:
        try:
            response = requests.get(ENDPOINT_URI)
            if response.status_code == 403:
                break
        # pylint: disable=bare-except
        except:
            pass
        # pylint: disable=bare-except
        timeout -= 0.1
        time.sleep(0.1)
    if timeout <= 0:
        raise RuntimeError("minio server didn't start")
    try:
        yield
    finally:
        process.terminate()
        process.wait()
    # pylint: enable=consider-using-with


def make_bucket(name):
    """Create a bucket."""
    session = botocore.session.get_session()
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
def s3(s3_base):
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
    def __init__(self, s3):
        name = f'bucket{S3.ID}'
        S3.ID += 1
        make_bucket(name)
        self.collection = S3Path(name).joinpath('collection')
        self.view = S3Path(name).joinpath('view')
        self.fs = s3

    # pylint: enable=redefined-outer-name
