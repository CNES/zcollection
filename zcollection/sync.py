# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Synchronization of concurrent accesses
======================================
"""
from __future__ import annotations

from typing import Callable
import abc
import threading

import fasteners


class Sync(abc.ABC):  # pragma: no cover
    """Interface of the classes handling the synchronization of concurrent
    accesses."""

    @abc.abstractmethod
    def __enter__(self) -> bool:
        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    @abc.abstractmethod
    def is_locked(self) -> bool:
        """Returns True if the lock is acquired, False otherwise."""


class NoSync(Sync):
    """This class is used when the user does not want to synchronize accesses
    to the collection, in other words, when there is no concurrency."""

    def __enter__(self) -> bool:
        return True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """As this class does not perform any synchronization, this method has
        nothing to do."""

    def is_locked(self) -> bool:
        """As this class does not perform any synchronization, this method
        always returns False."""
        return False


class ProcessSync(Sync):
    """This class is used when the user wants to synchronize accesses to the
    collection, in other words, when there is concurrency."""

    def __init__(self, path: str):
        self.lock = fasteners.InterProcessLock(path)

    def __enter__(self) -> bool:
        return self.lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.lock.release()
        except threading.ThreadError:
            pass

    def __reduce__(self) -> tuple[Callable, tuple[str]]:
        return (ProcessSync, (str(self.lock.path), ))

    def is_locked(self) -> bool:
        """Returns True if the lock is acquired, False otherwise."""
        return self.lock.exists()
