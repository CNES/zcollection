"""Local S3 server fixtures for tests and benchmarks.

Two backends, picked at runtime:

- :func:`minio_server` — a real ``minio`` binary, suitable for performance
  measurements. Skipped if ``minio`` is not on ``PATH`` (or
  ``MINIO_BIN`` environment variable does not point at one).

- :func:`moto_server` — pure-Python ``moto`` mock, suitable for
  correctness tests. Skipped if ``moto`` is not installed.

Both yield a small ``S3Endpoint`` carrying the URL, credentials, and a
helper that builds a ready-to-use :class:`zcollection.store.ObjectStore`
pointed at a fresh bucket.
"""

from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest


@dataclass(frozen=True, slots=True)
class S3Endpoint:
    """Connection details for a running local S3 server."""

    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"

    def object_store(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        read_only: bool = False,
    ) -> Any:
        """Build an :class:`ObjectStore` pointed at this endpoint.

        A unique bucket is created on demand when ``bucket`` is ``None``.
        """
        from zcollection.store.obstore_store import ObjectStore

        bucket = bucket or f"zc-{uuid.uuid4().hex[:12]}"
        self._ensure_bucket(bucket)
        url = f"s3://{bucket}"
        if prefix:
            url = f"{url}/{prefix.lstrip('/')}"
        return ObjectStore(
            url,
            client_options={"allow_http": True, "http1_only": True},
            credentials={
                "endpoint": self.endpoint,
                "access_key_id": self.access_key,
                "secret_access_key": self.secret_key,
                "region": self.region,
                "virtual_hosted_style_request": False,
            },
            read_only=read_only,
        )

    def _ensure_bucket(self, bucket: str) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover — opt-in dep
            pytest.skip(f"boto3 is required to seed test buckets: {exc}")
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )
        try:
            client.create_bucket(Bucket=bucket)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass
        except client.exceptions.BucketAlreadyExists:
            pass


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(url: str, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                resp.read(0)
                return
        except urllib.error.HTTPError:
            return  # any HTTP response means the server is up
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not start: {last_err!r}")


@contextmanager
def minio_server(
    data_dir: str, *, binary: str | None = None
) -> Iterator[S3Endpoint]:
    """Start a local MinIO server, yield its endpoint, stop it on exit.

    Args:
        data_dir: Directory the server stores its data in.
        binary: Explicit path to the ``minio`` executable. When ``None``,
            falls back to ``$MINIO_BIN`` and finally to ``minio`` on
            ``PATH``. The calling test is skipped if no binary is found.

    """
    binary = binary or os.environ.get("MINIO_BIN") or shutil.which("minio")
    if not binary:
        pytest.skip(
            "minio binary not found; pass --minio-bin, set MINIO_BIN, "
            "or install minio on PATH to run S3-backed tests.",
        )
    assert binary is not None
    port = _pick_free_port()
    console_port = _pick_free_port()
    access_key = "zcollection"
    secret_key = "zcollection-secret"
    env = {
        **os.environ,
        "MINIO_ROOT_USER": access_key,
        "MINIO_ROOT_PASSWORD": secret_key,
    }
    proc = subprocess.Popen(
        [
            binary,
            "server",
            data_dir,
            "--address",
            f"127.0.0.1:{port}",
            "--console-address",
            f"127.0.0.1:{console_port}",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    endpoint = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(f"{endpoint}/minio/health/ready", timeout=15.0)
        yield S3Endpoint(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


@contextmanager
def moto_server() -> Iterator[S3Endpoint]:
    """Start an in-process ``moto`` S3 mock, yield its endpoint."""
    try:
        from moto.server import ThreadedMotoServer
    except ImportError:
        pytest.skip(
            "moto[server] is not installed; "
            "install it to run S3 correctness tests.",
        )
    port = _pick_free_port()
    server = ThreadedMotoServer(port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(endpoint, timeout=10.0)
        yield S3Endpoint(
            endpoint=endpoint,
            access_key="testing",
            secret_key="testing",
        )
    finally:
        server.stop()
