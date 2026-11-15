# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partition callback protocols
============================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from collections.abc import Callable
import dataclasses

if TYPE_CHECKING:
    from .. import dataset
    from ..type_hints import ArrayLike

__all__ = ('MapCallable', 'PartitionCallable', 'PartitionInfo',
           'UpdateCallable')

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
    #: Partition path relative to the collection, formatted with ``/``.
    name: str
    #: The partitioned dimension.
    dimension: str
    #: Slice selecting the partition along the partitioned dimension within
    #: its overlap window. This also applies when the selected variables
    #: omit that dimension. Without overlap or during update inference,
    #: the slice is ``slice(None)``.
    indices: slice


#: pylint: disable=too-few-public-methods
class PartitionCallable(Protocol):
    """Protocol for partition callables.

    A partition callback receives a dataset and returns a result.
    """

    @property
    def __name__(self) -> str:
        """Name of the callable."""

        # Make checker happy.
        ...

    def __call__(self, zds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the partition function.

        Args:
            zds: The partition dataset.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The callback result.
        """


#: Alias for
#: :class:`~zcollection.collection.callable_objects.PartitionCallable`.
MapCallable = PartitionCallable


class UpdateCallable(Protocol):
    """Protocol for update callables.

    An update callback receives a dataset and returns a dictionary
    of arrays containing the updated values.
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
            zds: The partition dataset.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A dictionary of arrays with the updated values.
        """

        # Mandatory to make Pylance happy.
        ...
