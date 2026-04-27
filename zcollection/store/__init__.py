# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Store abstraction over Zarr v3 stores.

The factory :func:`open_store` selects an implementation from a URL:
``LocalStore`` (POSIX), ``MemoryStore`` (in-process), and ``IcechunkStore``
(transactional) are built in.
"""

from .base import Store, StoreSession
from .factory import open_store
from .layout import (
    CATALOG_DIR,
    IMMUTABLE_DIR,
    join_path,
    parent_path,
    relative_path,
)
from .local import LocalStore
from .memory import MemoryStore


def __getattr__(name: str):  # pragma: no cover — lazy optional dep
    if name == "IcechunkStore":
        from .icechunk_store import IcechunkStore

        return IcechunkStore
    raise AttributeError(name)


__all__ = (
    "CATALOG_DIR",
    "IMMUTABLE_DIR",
    "LocalStore",
    "MemoryStore",
    "Store",
    "StoreSession",
    "join_path",
    "open_store",
    "parent_path",
    "relative_path",
)
