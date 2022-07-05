# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Synchronization of concurrent accesses
======================================
"""
import abc


class Sync(abc.ABC):  # pragma: no cover
    """Interface of the classes handling the synchronization of concurrent
    accesses."""

    @abc.abstractmethod
    def __enter__(self) -> bool:
        ...

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...


class NoSync(Sync):
    """This class is used when the user does not want to synchronize accesses
    to the collection, in other words, when there is no concurrency."""

    def __enter__(self) -> bool:
        return True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """As this class does not perform any synchronization, this method has
        nothing to do."""
