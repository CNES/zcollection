# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Callable objects.
=================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from collections.abc import Callable
import dataclasses

if TYPE_CHECKING:
    from .. import dataset
    from ..type_hints import ArrayLike

#: Function type to load a partition or its overlap window, call an
#: :class:`UpdateCallable` on it and write the partition. It returns
#: :data:`zcollection.execution.SKIPPED` for an absent partition.
WrappedPartitionCallable = Callable[[str, tuple[Any, ...], dict[str, Any]],
                                    Any]


@dataclasses.dataclass(frozen=True)
class PartitionInfo:
    """Identify a partition and its position in the overlap window."""
    #: Partition key names and their values.
    partition: dict[str, int]
    #: Relative partition name, using the partitioner's formatting and '/'.
    name: str
    #: The partitioned dimension.
    dimension: str
    #: Slice of the partition along dimension within the overlap window,
    #: including when selected variables omit that dimension.
    #: Without overlap and during update inference, slice(None).
    indices: slice


#: pylint: disable=too-few-public-methods
class PartitionCallable(Protocol):
    """Protocol for partition callables.

    A partition callable is a function that accepts a dataset and
    returns a result.
    """

    @property
    def __name__(self) -> str:
        """Name of the callable."""

        # Make checker happy.
        ...

    def __call__(self, zds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the partition function.

        Args:
            zds: Dataset to partition.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the partition function.
        """


#: Alias for :class:`PartitionCallable`.
MapCallable = PartitionCallable


class UpdateCallable(Protocol):
    """Protocol for update callables.

    A callable update is a function that accepts a data set and returns
    a dictionary of arrays to update.
    """

    @property
    def __name__(self) -> str:
        """Name of the callable."""

        # Make checker happy.
        ...

    def __call__(self, zds: dataset.Dataset, *args,
                 **kwargs) -> dict[str, ArrayLike]:
        """Call the update function.

        Args:
            zds: Dataset to update.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dictionary of arrays to update.
        """

        # Mandatory to make Pylance happy.
        ...
