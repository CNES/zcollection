# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Async-runner and Dask glue for zcollection v3."""

from .runner import dask_map_async
from .scheduler import AsyncRunner, get_runner


__all__ = ("AsyncRunner", "dask_map_async", "get_runner")
