# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Synchronization of concurrent accesses
======================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import abc
import os
import threading

import fasteners

if TYPE_CHECKING:
    from collections.abc import Callable


class Sync(abc.ABC):  # pragma: no cover
    """Interface for synchronizing concurrent access."""

    @abc.abstractmethod
    def __enter__(self) -> bool:
        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    @abc.abstractmethod
    def is_locked(self) -> bool:
        """Return ``True`` if the lock is acquired, ``False`` otherwise."""


class NoSync(Sync):
    """Synchronizer that performs no locking."""

    def __enter__(self) -> bool:
        return True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit without performing synchronization."""

    def is_locked(self) -> bool:
        """Return ``False`` because no lock is used."""
        return False


class ProcessSync(Sync):
    """Synchronize access using an inter-process lock."""

    def __init__(self, path: str) -> None:
        self.lock = fasteners.InterProcessLock(path)

    def __enter__(self) -> bool:
        return self.lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.lock.release()
        except threading.ThreadError:
            pass

    def __reduce__(self) -> tuple[Callable, tuple[str]]:
        return (ProcessSync, (os.fsdecode(self.lock.path), ))

    def is_locked(self) -> bool:
        """Return whether this synchronizer holds its lock."""
        return self.lock.acquired
