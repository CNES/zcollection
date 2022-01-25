# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning scheme.
====================
"""
from __future__ import annotations

import abc
import collections
import re
from typing import (Any, ClassVar, Dict, Iterator, Optional, Sequence, Tuple,
                    Union)

import dask.array
import numpy

from .. import dataset
from ..typing import NDArray

#: Object that can be used as a numpy array
Array = Union[dask.array.Array, NDArray]

#: Object that represents a partitioning scheme
Partition = Tuple[Tuple[Tuple[str, Any], ...], slice]

#: Allowed data types for partitioning schemes
DATA_TYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
              "uint64")


def _logical_or_reduce(
    arr: dask.array.Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> dask.array.Array:
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

    return dask.array.reduction(
        arr[1:] != arr[:-1],  # type: ignore
        chunk=chunk,
        aggregate=aggregate,
        axis=axis,
        keepdims=False,
        dtype=numpy.bool_)


def unique(arr: dask.array.Array) -> Tuple[NDArray, NDArray]:
    """
    Return unique elements and their indices.

    Args:
        arr: Array of elements.

    Returns:
        Tuple of unique elements and their indices.
    """
    size = arr.shape[0]
    chunks = arr.chunks[0]
    #: pylint: disable=not-callable
    mask = dask.array.empty((size, ), dtype=numpy.bool_, chunks=chunks)
    #: pylint: enable=not-callable
    mask[0] = True
    mask[1:] = (_logical_or_reduce(arr, axis=1)
                if arr.ndim > 1 else arr[1:] != arr[:-1])
    dtype = numpy.uint32 if size < 2**32 else numpy.uint64
    indices = dask.array.arange(size, dtype=dtype, chunks=chunks)
    mask = mask.persist()
    return arr[mask].compute(), indices[mask].compute()


def difference(arr: NDArray) -> NDArray:
    """
    Calculate the difference between each element in the array and the
    previous element.

    Args:
        arr: Array to calculate the difference for.

    Returns:
        Array of differences
    """
    return arr[1:] - arr[:-1]  # type: ignore


def concatenate_item(arr: NDArray, item: Any) -> NDArray:
    """
    Concatenate an array with a given item.

    Args:
        arr: Array to concatenate.
        item: Item to concatenate.

    Returns:
        Concatenated array.
    """
    return numpy.concatenate([arr, numpy.array([item], dtype=arr.dtype)])


class Partitioning(abc.ABC):
    """Partitioning scheme

    Args:
        variables:  List of variables to be used for partitioning
        dtype: The list of data types allowing to store the values of variables
            in a binary representation without loss of information.
            Defaults to int64.
    """
    __slots__ = ("_dtype", "_pattern", "variables")

    #: The ID of the partitioning scheme
    ID: ClassVar[Optional[str]] = None

    def __init__(self,
                 variables: Sequence[str],
                 dtype: Optional[Sequence[str]] = None) -> None:
        if isinstance(dtype, str):
            raise TypeError("dtype must be a sequence of strings")
        #: Variables to be used for the partitioning.
        self.variables = tuple(variables)
        #: Data type used to store variable values in a binary representation
        #:  without data loss.
        self._dtype = dtype or ("int64", ) * len(self.variables)
        #: The regular expression that matches the partitioning scheme.
        self._pattern = self._regex().search

        if len(set(self._dtype) - set(DATA_TYPES)) != 0:
            raise ValueError(
                f"Data type must be one of {', '.join(DATA_TYPES)}.")

    def dtype(self) -> Tuple[Tuple[str, str], ...]:
        """Return the data type of the partitioning scheme."""
        return tuple(zip(self._keys(), self._dtype))

    def _keys(self) -> Sequence[str]:
        """Return the different keys of a partition."""
        return self.variables

    def _regex(self) -> re.Pattern:
        """Return a regular expression that matches the partitioning scheme."""
        return re.compile(".".join(
            (f"({item})=(.*)" for item in self._keys())))

    @abc.abstractmethod
    def _split(
        self,
        variables: Dict[str, dask.array.Array],
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
        ...  # pragma: no cover

    @staticmethod
    def _partition(selection: Tuple[Tuple[str, Any], ...]) -> Tuple[str, ...]:
        """Format the partitioning scheme"""
        return tuple(f"{k}={v}" for k, v in selection)

    def index_dataset(self, ds: dataset.Dataset) -> Iterator[Partition]:
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
            (name, ds.variables[name].raw_data) for name in self.variables)
        # If the dask array is too chunked, the calculation is excessively
        # long.
        for item, array in variables.items():
            variables[item] = array.rechunk().persist()
        return self._split(variables)

    def split_dataset(
        self,
        ds: dataset.Dataset,
        axis: str,
    ) -> Iterator[Tuple[Tuple[str, ...], Dict[str, slice]]]:
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
            if len(ds.variables[item].raw_data.shape) != 1:
                raise ValueError(f"f{item!r} must be a one-dimensional array")
        return ((self._partition(selection), {
            axis: indexer
        }) for selection, indexer in self.index_dataset(ds))

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the partitioning scheme.

        Returns:
            The configuration of the partitioning scheme.
        """
        config = dict(id=self.ID)
        slots = (getattr(_class, "__slots__", ())
                 for _class in reversed(self.__class__.__mro__))
        config.update((attr, getattr(self, attr)) for _class in slots
                      for attr in _class if not attr.startswith("_"))
        return config

    @classmethod
    def from_config(cls, config) -> "Partitioning":
        """Create a partitioning scheme from a configuration.

        Args:
            config: The configuration of the partitioning scheme.

        Returns:
            The partitioning scheme.
        """
        return cls(**config)

    def parse(self, partition: str) -> Tuple[Tuple[str, int], ...]:
        """Parse a partitioning scheme.

        Args:
            partition: The partitioning scheme to be parsed.

        Returns:
            The parsed partitioning scheme.
        """
        match = self._pattern(partition)
        if match is None:
            raise ValueError(
                f"Partition is not driven by this instance: {partition}")
        groups = match.groups()
        return tuple((groups[ix], int(groups[ix + 1]))
                     for ix in range(0, len(groups), 2))

    @staticmethod
    def join(partition_scheme: Tuple[Tuple[str, int], ...], sep: str) -> str:
        """Join a partitioning scheme.

        Args:
            partition_scheme: The partitioning scheme to be joined.
            sep: The separator to be used.

        Returns:
            The joined partitioning scheme.
        """
        return sep.join(f"{k}={v}" for k, v in partition_scheme)

    def values(self, partition: str) -> Tuple[Tuple[str, Any], ...]:
        """Return the values of the partitioning scheme.

        Args:
            partition: The partitioning scheme to be parsed.

        Returns:
            The values of the keys constituting the partitioning scheme.
        """
        return self.parse(partition)
