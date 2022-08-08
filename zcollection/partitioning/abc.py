# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning scheme.
====================
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterator, Sequence, Tuple
import abc
import collections
import re

import dask.array
import dask.array.core
import dask.array.creation
import dask.array.reductions
import dask.array.wrap
import fsspec
import numpy

from .. import dataset
from ..typing import NDArray

#: Object that represents a partitioning scheme
Partition = Tuple[Tuple[Tuple[str, Any], ...], slice]

#: Allowed data types for partitioning schemes
DATA_TYPES = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
              'uint64')


def _logical_or_reduce(
    arr: dask.array.core.Array,
    axis: int | tuple[int, ...] | None = None,
) -> dask.array.core.Array:
    """Implementation of `numpy.logical_or` reduction with dask.

    Args:
        arr: Array to reduce.
        axis: Axis to reduce. If this is None, a reduction is performed over
            all the axes. If this is a tuple of ints, a reduction is performed
            on multiple axes, instead of a single axis or all the axes as
            before.

    Returns:
        Reduced array.
    """
    axis = axis or 0

    #: pylint: disable=unused-argument
    # The function signature is required by the `dask.array.reduce` function.
    def chunk(block, axis, keepdims):
        return block

    def aggregate(block, axis, keepdims):
        return numpy.logical_or.reduce(block, axis=axis)

    #: pylint: enable=unused-argument

    return dask.array.reductions.reduction(
        arr[1:] != arr[:-1],  # type: ignore
        chunk=chunk,
        aggregate=aggregate,
        axis=axis,
        keepdims=False,
        dtype=numpy.bool_)


def unique(arr: dask.array.core.Array) -> tuple[NDArray, NDArray]:
    """Return unique elements and their indices.

    Args:
        arr: Array of elements.

    Returns:
        Tuple of unique elements and their indices.
    """
    size = arr.shape[0]
    chunks = arr.chunks[0]
    #: pylint: disable=not-callable
    mask = dask.array.wrap.empty((size, ), dtype=numpy.bool_, chunks=chunks)
    #: pylint: enable=not-callable
    mask[0] = True
    mask[1:] = (_logical_or_reduce(arr, axis=1)
                if arr.ndim > 1 else arr[1:] != arr[:-1])
    dtype = numpy.uint32 if size < 2**32 else numpy.uint64
    indices = dask.array.creation.arange(size, dtype=dtype, chunks=chunks)
    mask = mask.persist()
    return arr[mask].compute(), indices[mask].compute()


def difference(arr: NDArray) -> NDArray:
    """Calculate the difference between each element in the array and the
    previous element.

    Args:
        arr: Array to calculate the difference for.

    Returns:
        Array of differences
    """
    return arr[1:] - arr[:-1]  # type: ignore


def concatenate_item(arr: NDArray, item: Any) -> NDArray:
    """Concatenate an array with a given item.

    Args:
        arr: Array to concatenate.
        item: Item to concatenate.

    Returns:
        Concatenated array.
    """
    return numpy.concatenate([arr, numpy.array([item], dtype=arr.dtype)])


def list_partitions(
    fs: fsspec.AbstractFileSystem,
    path: str,
    depth: int,
    root: bool = True,
) -> Iterator[str]:
    """The number of variables used for partitioning.

    The function will go down the tree and return all the files present when the
    requested depth is reached.

    Args:
        fs: file system object
        path: path to the directory
        depth: maximum depth of the directory tree.
        root: if True, the path is the root of the tree.

    Returns:
        Iterator of (path, directories, files).
    """
    if depth == -1:
        return

    if depth == 0:
        yield from sorted(fs.ls(path, detail=False))
    elif root:
        folders = map(
            lambda info: info['name'].rstrip('/'),
            filter(lambda info: info['type'] == 'directory',
                   fs.ls(path, detail=True)))
        for pathname in sorted(folders):
            yield from list_partitions(fs,
                                       pathname,
                                       depth=depth - 1,
                                       root=False)
    else:
        for item in sorted(fs.ls(path, detail=False)):
            yield from list_partitions(fs, item, depth=depth - 1, root=False)


class Partitioning(abc.ABC):
    """Partitioning scheme.

    Args:
        variables:  List of variables to be used for partitioning
        dtype: The list of data types allowing to store the values of variables
            in a binary representation without loss of information.
            Defaults to int64.
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
        self._dtype = dtype or ('int64', ) * len(self.variables)
        #: The regular expression that matches the partitioning scheme.
        self._pattern = self._regex().search

        if len(set(self._dtype) - set(DATA_TYPES)) != 0:
            raise ValueError(
                f"Data type must be one of {', '.join(DATA_TYPES)}.")

    def __len__(self) -> int:
        """Return the number of partitions."""
        return len(self._dtype)

    def dtype(self) -> tuple[tuple[str, str], ...]:
        """Return the data type of the partitioning scheme."""
        return tuple(zip(self._keys(), self._dtype))

    def _keys(self) -> Sequence[str]:
        """Return the different keys of a partition."""
        return self.variables

    def _regex(self) -> re.Pattern:
        """Return a regular expression that matches the partitioning scheme."""
        return re.compile('.'.join(f'({item})=(.*)' for item in self._keys()))

    @abc.abstractmethod
    def _split(
        self,
        variables: dict[str, dask.array.core.Array],
    ) -> Iterator[Partition]:
        """Split the variables constituting the partitioning into partitioning
        schemes.

        Args:
            variables: The variables to be split constituting the
                partitioning scheme.

        Returns:
            A sequence of tuples that contains the partitioning
                scheme and the associated indexer to divide the dataset on each
                partition found..
        """

    @staticmethod
    def _partition(selection: tuple[tuple[str, Any], ...]) -> tuple[str, ...]:
        """Format the partitioning scheme."""
        return tuple(f'{k}={v}' for k, v in selection)

    def index_dataset(
        self,
        ds: dataset.Dataset,
    ) -> Iterator[Partition]:
        """Yield the indexing scheme for the given dataset.

        Args:
            ds: The dataset to be indexed.

        Yields:
            The indexing scheme for the partitioning scheme.

        Raises:
            ValueError: if one of the variables needs for the partitioning
                is not monotonic.
        """
        variables = collections.OrderedDict(
            (name, ds.variables[name].array) for name in self.variables)
        # If the dask array is too chunked, the calculation is excessively
        # long.
        return self._split(
            {name: arr.rechunk().persist()
             for name, arr in variables.items()})

    def split_dataset(
        self,
        ds: dataset.Dataset,
        axis: str,
    ) -> Iterator[tuple[tuple[str, ...], dict[str, slice]]]:
        """Split the dataset into partitions.

        Args:
            ds:  The dataset to be split.
            axis: The axis to be used for the splitting.

        Yields:
            The partitioning scheme and the indexer to divide the dataset on
            each partition found.

        Raises:
            ValueError: if one of the variables needs for the partitioning
                is not a one-dimensional array.
        """
        for item in self.variables:
            if len(ds.variables[item].shape) != 1:
                raise ValueError(f'f{item!r} must be a one-dimensional array')
        return ((self._partition(selection), {
            axis: indexer
        }) for selection, indexer in self.index_dataset(ds))

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of the partitioning scheme.

        Returns:
            The configuration of the partitioning scheme.
        """
        config = dict(id=self.ID)
        slots = (getattr(_class, '__slots__', ())
                 for _class in reversed(self.__class__.__mro__))
        config.update((attr, getattr(self, attr)) for _class in slots
                      for attr in _class if not attr.startswith('_'))
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
        """Parse a partitioning scheme.

        Args:
            partition: The partitioning scheme to be parsed.

        Returns:
            The parsed partitioning scheme.
        """
        match = self._pattern(partition)
        if match is None:
            raise ValueError(
                f'Partition is not driven by this instance: {partition}')
        groups = match.groups()
        return tuple((groups[ix], int(groups[ix + 1]))
                     for ix in range(0, len(groups), 2))

    @abc.abstractmethod
    def encode(
        self,
        partition: tuple[tuple[str, int], ...],
    ) -> tuple[Any, ...]:
        """Encode a partitioning scheme to the handled values.

        Args:
            partition: The partitioning scheme to be encoded.

        Returns:
            The encoded partitioning scheme.
        """

    @abc.abstractmethod
    def decode(self, values: tuple[Any, ...]) -> tuple[tuple[str, int], ...]:
        """Decode a partitioning scheme.

        Args:
            values: The encoded partitioning scheme.

        Returns:
            The decoded partitioning scheme.
        """

    @staticmethod
    def join(partition_scheme: tuple[tuple[str, int], ...], sep: str) -> str:
        """Join a partitioning scheme.

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
            fs: The filesystem to be used.
            path: The path to the directory containing the partitions.

        Yields:
            The partitions.
        """
        return list_partitions(fs, path, depth=len(self) - 1)
