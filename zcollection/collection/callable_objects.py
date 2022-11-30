"""
Callable objects.
=================
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

from .. import dataset
from ..type_hints import ArrayLike

#: Function type to load and call a callback function of type
#: :class:`PartitionCallable`.
WrappedPartitionCallable = Callable[[Sequence[str]], None]


#: pylint: disable=too-few-public-methods
class PartitionCallable(Protocol):
    """Protocol for partition callables.

    A partition callable is a function that accepts a dataset and
    returns a result.
    """

    @property
    def __name__(self) -> str:
        """Name of the callable."""
        # pylint: disable=unnecessary-ellipsis
        # Make checker happy.
        ...
        # pylint: enable=unnecessary-ellipsis

    def __call__(self, ds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the partition function.

        Args:
            ds: Dataset to partition.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the partition function.
        """


class MapCallable(Protocol):
    """Protocol for map callables.

    A callable map is a function that accepts a data set, a list of
    arguments, a dictionary of keyword arguments and returns a result.
    """

    def __call__(self, ds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the map function.

        Args:
            ds: Dataset to map.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the map function.
        """

    #: pylint: enable=too-few-public-methods


class UpdateCallable(Protocol):
    """Protocol for update callables.

    A callable update is a function that accepts a data set and returns
    a dictionary of arrays to update.
    """

    @property
    def __name__(self) -> str:
        """Name of the callable."""
        # pylint: disable=unnecessary-ellipsis
        # Make checker happy.
        ...
        # pylint: enable=unnecessary-ellipsis

    def __call__(self, ds: dataset.Dataset, *args,
                 **kwargs) -> dict[str, ArrayLike]:
        """Call the update function.

        Args:
            ds: Dataset to update.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dictionary of arrays to update.
        """
        # pylint: disable=unnecessary-ellipsis
        # Mandatory to make Pylance happy.
        ...
        # pylint: enable=unnecessary-ellipsis
