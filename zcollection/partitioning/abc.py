# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning scheme.
====================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional
import abc
import collections
from collections.abc import Callable, Generator, Iterator, Sequence
import operator
import re
from re import Match

import dask.array.core
import dask.array.creation
import dask.array.reductions
import dask.array.wrap
import numpy

if TYPE_CHECKING:
    import fsspec

    from .. import dataset
    from ..type_hints import ArrayLike, DTypeLike, NDArray

#: Type representing a partitioning scheme.
Partition = tuple[tuple[tuple[str, Any], ...], slice]

#: Type of callable used to parse partition keys.
PatternType = Callable[[str], Optional[Match[str]]]

#: Allowed data types for partition keys.
DATA_TYPES = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
              'uint64')


def _logical_or_reduce(
    arr: dask.array.core.Array,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> dask.array.core.Array:
    """Reduce changes between consecutive rows using logical OR.

    The function compares ``arr[1:]`` with ``arr[:-1]`` and reduces the
    resulting Boolean array along ``axis``, which defaults to 0.

    Args:
        arr: The array to compare.
        axis: The axis or axes over which to reduce. Defaults to 0.

    Returns:
        The logical-OR reduction of the comparison mask.
    """
    axis = axis or 0

    #: pylint: disable=unused-argument
    # The chunk and aggregate signatures are required by Dask reductions.
    def chunk(block, axis, keepdims) -> Any:
        return block

    def aggregate(block, axis, keepdims) -> Any:
        return numpy.logical_or.reduce(block, axis=axis)

    #: pylint: enable=unused-argument

    return dask.array.reductions.reduction(arr[1:] != arr[:-1],
                                           chunk=chunk,
                                           aggregate=aggregate,
                                           axis=axis,
                                           keepdims=False,
                                           dtype=numpy.bool_)


def unique(arr: dask.array.core.Array) -> tuple[NDArray, NDArray]:
    """Return the first element and index of each run of equal values.

    Args:
        arr: The input array.

    Returns:
        The first values of consecutive runs and their indices.
    """
    size: int = arr.shape[0]
    chunks: tuple[int, ...] = arr.chunks[0]
    #: pylint: disable=not-callable
    mask: dask.array.core.Array = dask.array.wrap.empty((size, ),
                                                        dtype=numpy.bool_,
                                                        chunks=chunks)
    #: pylint: enable=not-callable
    mask[0] = True
    mask[1:] = (_logical_or_reduce(arr, axis=1)
                if arr.ndim > 1 else arr[1:] != arr[:-1])
    dtype: DTypeLike = numpy.uint32 if size < 2**32 else numpy.uint64
    indices: dask.array.core.Array = dask.array.creation.arange(
        size,
        dtype=dtype,
        chunks=chunks,  # type: ignore[arg-type]
    )
    mask = mask.persist()
    return arr[mask].compute(), indices[mask].compute()


def unique_and_check_monotony(arr: ArrayLike) -> tuple[NDArray, NDArray]:
    """Return unique values and their first indices, checking monotonicity.

    Args:
        arr: The input array.

    Returns:
        The unique values and indices of their first occurrences.

    Raises:
        ValueError: If the values are not monotonic.
    """
    index: NDArray
    indices: NDArray

    index, indices = numpy.unique(arr, axis=0, return_index=True)
    if not numpy.all(numpy.diff(indices) > 0):
        raise ValueError('index is not monotonic')
    return index, indices


def difference(arr: NDArray) -> NDArray:
    """Return the differences between consecutive array elements.

    Args:
        arr: Array to calculate the difference for.

    Returns:
        The differences between consecutive elements.
    """
    return arr[1:] - arr[:-1]


def concatenate_item(arr: NDArray, item: Any) -> NDArray:
    """Concatenate an array with a given item.

    Args:
        arr: The input array.
        item: The item to append.

    Returns:
        Concatenated array.
    """
    return numpy.concatenate([arr, numpy.array([item], dtype=arr.dtype)])


def list_partitions(
    fs: fsspec.AbstractFileSystem,
    path: str,
    depth: int,
    *,
    root: bool = True,
) -> Iterator[str]:
    """Yield paths at a specified depth of a directory tree.

    Traverse the tree up to ``depth``. At the first level, only directories
    are considered, and names beginning with a dot are skipped.

    Args:
        fs: The file system to traverse.
        path: The starting directory.
        depth: The maximum depth to traverse.
        root: Whether ``path`` is the root of the directory tree.

    Yields:
        Paths as strings.
    """
    if depth == -1:
        return StopIteration()  # type: ignore[return-value]

    if root:
        folders = (info['name'].rstrip('/') for info in filter(
            lambda info: info['type'] == 'directory' and not info['name'].
            split(fs.sep)[-1].startswith('.'),
            fs.ls(path, detail=True),
        ))
        # If we're partitioning at top level (example: by year)
        if depth == 0:
            yield from folders
            return StopIteration()  # type: ignore[return-value]

        for pathname in folders:
            yield from list_partitions(fs,
                                       pathname,
                                       depth=depth - 1,
                                       root=False)
        return StopIteration()  # type: ignore[return-value]

    if depth == 0:
        yield from fs.ls(path, detail=False)
        return StopIteration()  # type: ignore[return-value]

    for item in fs.ls(path, detail=False):
        yield from list_partitions(fs, item, depth=depth - 1, root=False)
    return StopIteration()  # type: ignore[return-value]


class Partitioning(metaclass=abc.ABCMeta):
    """Define a partitioning scheme using one or more variables.

    Args:
        variables: Names of the partitioning variables.
        dtype: Optional sequence of data types used to store partition keys.
            Supported types are ``int8``, ``int16``, ``int32``, ``int64``,
            ``uint8``, ``uint16``, ``uint32``, ``uint64``.
            Defaults to ``int64`` for all variables.

    Raises:
        TypeError: If ``dtype`` is a string instead of a sequence.
        ValueError: If no partitioning variables are given or a data type
            is not supported.
    """
    __slots__ = ('_dtype', '_pattern', 'variables')

    #: The ID of the partitioning scheme
    ID: ClassVar[str | None] = None

    def __init__(self,
                 variables: Sequence[str],
                 dtype: Sequence[str] | None = None) -> None:
        if isinstance(dtype, str):
            raise TypeError('dtype must be a sequence of strings')
        if len(variables) == 0:
            raise ValueError('variables must not be empty')
        #: Variables to be used for the partitioning.
        self.variables = tuple(variables)
        #: Data type used to store variable values in a binary representation
        #: without data loss.
        self._dtype: tuple[str, ...] = tuple(
            dtype) if dtype is not None else ('int64', ) * len(self.variables)
        #: The regular expression that matches the partitioning scheme.
        self._pattern: PatternType = self._regex().search

        if len(set(self._dtype) - set(DATA_TYPES)) != 0:
            raise ValueError(
                f"Data type must be one of {', '.join(DATA_TYPES)}.")

    def __len__(self) -> int:
        """Return the number of partitioning keys."""
        return len(self._dtype)

    def dtype(self) -> tuple[tuple[str, str], ...]:
        """Return the data type of the partitioning scheme."""
        return tuple(zip(self._keys(), self._dtype, strict=True))

    def _keys(self) -> Sequence[str]:
        """Return the names of the partitioning keys."""
        return self.variables

    def _regex(self) -> re.Pattern:
        """Return a regular expression that matches the partitioning scheme."""
        return re.compile('.'.join(f'({item})=(.*)' for item in self._keys()))

    @abc.abstractmethod
    def _split(
        self,
        variables: dict[str, NDArray],
    ) -> Iterator[Partition]:
        """Split partitioning variable values into partition keys and indexers.

        Args:
            variables: Values of the partitioning variables.

        Returns:
            Partition keys and the corresponding dataset indexers.
        """

    @staticmethod
    def _partition(selection: tuple[tuple[str, Any], ...]) -> tuple[str, ...]:
        """Format a partition key."""
        return tuple(f'{k}={v}' for k, v in selection)

    def index_dataset(
        self,
        zds: dataset.Dataset,
    ) -> Iterator[Partition]:
        """Yield the partition keys and indexers of a dataset.

        Args:
            zds: The dataset to partition.

        Yields:
            Each partition key and the indexer selecting its rows.

        Raises:
            ValueError: If a partitioning variable is not monotonic.
        """
        variables = collections.OrderedDict(
            (name, zds.variables[name].array) for name in self.variables)
        if zds.delayed:
            # Rechunk very small Dask chunks to limit processing overhead.
            return self._split({
                name: arr.rechunk().persist()
                for name, arr in variables.items()
            })
        return self._split(variables)

    def split_dataset(
        self,
        zds: dataset.Dataset,
        axis: str,
    ) -> Iterator[tuple[tuple[str, ...], dict[str, slice]]]:
        """Split the dataset into partitions.

        Args:
            zds: The dataset to split.
            axis: The variable used as the collection axis.

        Yields:
            Each partition key and the indexer selecting its rows.

        Raises:
            ValueError: If a partitioning variable is not one-dimensional.
        """
        for item in self.variables:
            if len(zds.variables[item].shape) != 1:
                raise ValueError(f'{item!r} must be a one-dimensional array')
        return ((self._partition(selection), {
            axis: indexer
        }) for selection, indexer in self.index_dataset(zds))

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of the partitioning scheme.

        Returns:
            The configuration of the partitioning scheme.
        """
        config: dict[str, str | tuple[str, ...] | None] = {'id': self.ID}
        slots: Generator[tuple[str, ...]] = (getattr(
            _class, '__slots__',
            ()) for _class in reversed(self.__class__.__mro__))
        config.update((attr, getattr(self, attr)) for _class in slots
                      for attr in _class if not attr.startswith('_'))
        config['dtype'] = self._dtype
        return config

    @classmethod
    def from_config(cls, config) -> Partitioning:
        """Create a partitioning scheme from a configuration.

        Args:
            config: The configuration of the partitioning scheme.
        Returns:
            The partitioning scheme.
        """
        return cls(**config)

    def parse(self, partition: str) -> tuple[tuple[str, int], ...]:
        """Parse a partition path into its keys.

        Args:
            partition: The partition path to parse.

        Returns:
            The partition keys and their values.

        Raises:
            ValueError: If the path does not match this partitioning scheme.
        """
        match: Match[str] | None = self._pattern(partition)
        if match is None:
            raise ValueError(
                f'Partition is not driven by this instance: {partition}')
        groups: tuple[str, ...] = match.groups()
        return tuple(zip(groups[::2], map(int, groups[1::2]), strict=True))

    @abc.abstractmethod
    def encode(
        self,
        partition: tuple[tuple[str, int], ...],
    ) -> tuple[Any, ...]:
        """Encode partition keys into values of the declared data types.

        Args:
            partition: The partitioning scheme to be encoded.

        Returns:
            The encoded partitioning scheme.
        """

    @abc.abstractmethod
    def decode(self, values: tuple[Any, ...]) -> tuple[tuple[str, int], ...]:
        """Decode values into partition keys.

        Args:
            values: The encoded partitioning scheme.

        Returns:
            The decoded partitioning scheme.
        """

    @staticmethod
    def join(partition_scheme: tuple[tuple[str, int], ...], sep: str) -> str:
        """Join partition keys into a path.

        Args:
            partition_scheme: The partitioning scheme to be joined.
            sep: The separator to be used.

        Returns:
            The joined partitioning scheme.
        """
        return sep.join(f'{k}={v}' for k, v in partition_scheme)

    def list_partitions(
        self,
        fs: fsspec.AbstractFileSystem,
        path: str,
    ) -> Iterator[str]:
        """List the partitions.

        Args:
            fs: The file system containing the partitions.
            path: The base directory containing the partitions.

        Yields:
            The partitions, ordered by the values of their keys. The values
            are the ones :meth:`parse` reads, so ``cycle=2`` comes before
            ``cycle=10``.
        """
        parsed_partitions = []

        for partition in list_partitions(fs, path, depth=len(self) - 1):
            try:
                parsed_partition = self.parse(partition)
            except ValueError:
                continue

            parsed_partitions.append((parsed_partition, partition))

        parsed_partitions.sort(key=operator.itemgetter(0))

        for _, partition in parsed_partitions:
            yield partition
