# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset variable.
=================
"""
from __future__ import annotations

from typing import Any, Iterator, Sequence, TypeVar
import abc
import collections

import dask.array.core
import dask.base
import dask.threaded
import numcodecs.abc
import numpy
import xarray
import zarr

from .. import mathematics, meta, representation
from ..meta import Attribute
from ..type_hints import NDArray, NDMaskedArray

#: The dask array getter used to access the data.
GETTER = dask.array.core.getter

#: Generic type for a dataset variable.
T = TypeVar('T', bound='Variable')


class ModifiedVariableError(RuntimeError):
    """Raised when a variable has been modified since is was initialized."""

    def __str__(self) -> str:
        """Get the string representation of the exception.

        Returns:
            The string representation of the exception.
        """
        return ('You tried to access the data of a variable that has been '
                'modified since its initialization. Try to re-load the '
                'dataset.')


def _variable_repr(var: Variable) -> str:
    """Get the string representation of a variable.

    Args:
        var: A variable.

    Returns:
        The string representation of the variable.
    """
    # Dimensions
    dims_str = representation.dimensions(dict(zip(var.dimensions, var.shape)))
    lines = [
        f'<{var.__module__}.{var.__class__.__name__} {dims_str}>',
        f'{var.data!r}'
    ]
    # Attributes
    if len(var.attrs):
        lines.append('  Attributes:')
        lines += representation.attributes(var.attrs)
    # Filters
    if var.filters:
        lines.append('  Filters:')
        lines += [f'    {item!r}' for item in var.filters]
    # Compressor
    if var.compressor:
        lines.append('  Compressor:')
        lines += [f'    {var.compressor!r}']
    return '\n'.join(lines)


def _blockdims_from_blockshape(
        shape: tuple[int, ...],
        chunks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Convert a blockshape to a blockdims tuple."""
    return tuple(((chunk_item, ) * (shape_item // chunk_item) +
                  ((shape_item % chunk_item, ) if shape_item %
                   chunk_item else ()) if shape_item else (0, ))
                 for shape_item, chunk_item in zip(shape, chunks))


def from_zarr_array(
    array: zarr.Array,
    shape: Sequence[int],
    chunks: Sequence[int],
    name: str,
    lock: bool = False,
    asarray=True,
    inline_array=True,
) -> dask.array.core.Array:
    """Create a dask array from a zarr array.

    Args:
        array: A zarr array.

    Returns:
        The dask array.
    """
    normalized_chunks = sum(
        (_blockdims_from_blockshape(
            (shape_item, ),
            (chunk_item, )) if not isinstance(chunk_item, (tuple, list)) else
         (chunk_item, ) for shape_item, chunk_item in zip(shape, chunks)),
        (),
    )
    dsk = dask.array.core.graph_from_arraylike(
        array,
        normalized_chunks,
        shape,
        name,
        getitem=GETTER,
        lock=lock,
        asarray=asarray,
        dtype=array.dtype,
        inline_array=inline_array,
    )
    return dask.array.core.Array(dsk,
                                 name,
                                 normalized_chunks,
                                 meta=array,
                                 dtype=array.dtype)


class Variable:
    """Variables hold multi-dimensional arrays of data.

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """
    __slots__ = ('tableau', 'attrs', 'compressor', 'dimensions', 'fill_value',
                 'filters', 'name')

    def __init__(self,
                 name: str,
                 data: Any,
                 dimensions: Sequence[str],
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        #: Variable name
        self.name = name
        #: Variable data as a dask array.
        self.tableau: Any = data
        #: Variable dimensions
        self.dimensions = dimensions
        #: Variable attributes
        self.attrs: Sequence[Attribute] = attrs or tuple()
        #: Compressor used to compress the data during writing data to disk
        self.compressor = compressor
        #: Value to use for uninitialized values
        self.fill_value = fill_value
        #: Filters to apply before writing data to disk
        self.filters = filters

    @property
    @abc.abstractmethod
    def TABLEAU(self) -> dask.array.core.Array:
        """Variable data as a dask array."""

    @property
    @abc.abstractmethod
    def data(self) -> dask.array.core.Array:
        """Return the underlying dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """

    @data.setter
    @abc.abstractmethod
    def data(self, data: Any) -> None:
        """Defines the underlying dask array. If the data provided is a masked
        array, it's converted to an array, where the masked values are replaced
        by its fill value, and its fill value becomes the new fill value of
        this instance. Otherwise, the underlying array is defined as the new
        data and the fill value is set to None.

        Args:
            data: The new data to use

        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """

    @property
    @abc.abstractmethod
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """

    @property
    @abc.abstractmethod
    def dtype(self) -> numpy.dtype:
        """Return the data type of the variable."""

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the variable."""
        return len(self.dimensions)

    @property
    def size(self: Any) -> int:
        """Return the size of the variable."""
        return mathematics.prod(self.shape)

    @property
    def nbytes(self):
        """Return the number of bytes used by the variable."""
        return self.size * self.dtype.itemsize

    @abc.abstractmethod
    def persist(self: T, **kwargs) -> T:
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.persist`.

        Returns:
            The variable
        """

    @abc.abstractmethod
    def compute(self, **kwargs) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.compute`.
        """

    @abc.abstractmethod
    def fill(self: T) -> T:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """

    @abc.abstractmethod
    def duplicate(self: T, data: Any) -> T:
        """Create a new variable from the properties of this instance and the
        data provided.

        Args:
            data: Variable data.

        Returns:
            New variable.

        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """

    @classmethod
    @abc.abstractmethod
    def from_zarr(cls: type[T], array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> T:
        """Create a new variable from a zarr array.

        Args:
            array: The zarr array
            name: Name of the variable
            dimension: Name of the attribute that defines the dimensions of the
                variable
            **kwargs: Keyword arguments passed to
                :func:`dask.array.from_array`

        Returns:
            The variable
        """

    @abc.abstractmethod
    def concat(self: T, other: T | Sequence[T], dim: str) -> T:
        """Concatenate this variable with another variable or a list of
        variables along a dimension.

        Args:
            other: Variable or list of variables to concatenate with this
                variable.
            dim: Dimension to concatenate along.

        Returns:
            New variable.

        Raises:
            ValueError: if the variables provided is an empty sequence.
        """

    @abc.abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.
        Returns:
            The variable slice.
        """

    @abc.abstractmethod
    def isel(self: T, key: tuple[slice, ...]) -> T:
        """Return a new variable with data selected along the given dimension
        indices.

        Args:
            key: Dimension indices to select

        Returns:
            The new variable
        """

    @abc.abstractmethod
    def set_for_insertion(self: T) -> T:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """

    @abc.abstractmethod
    def rename(self: T, name: str) -> T:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """

    @abc.abstractmethod
    def rechunk(self: T, **kwargs) -> T:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """

    def fill_attrs(self, var: meta.Variable) -> None:
        """Fill the variable attributes using the provided metadata.

        Args:
            var: Variable's metadata.
        """
        self.attrs = var.attrs

    def metadata(self) -> meta.Variable:
        """Get the variable metadata.

        Returns:
            Variable metadata
        """
        return meta.Variable(self.name, self.dtype, self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def have_same_properties(self, other: Variable) -> bool:
        """Return true if this instance and the other variable have the same
        properties."""
        return self.metadata() == other.metadata()

    def dimension_index(self) -> Iterator[tuple[str, int]]:
        """Return an iterator over the variable dimensions and their index.

        Returns:
            An iterator over the variable dimensions
        """
        yield from ((item, ix) for ix, item in enumerate(self.dimensions))

    def to_xarray(self) -> xarray.Variable:
        """Convert the variable to an xarray.Variable.

        Returns:
            Variable as an xarray.Variable
        """
        encoding = {}
        if self.filters:
            encoding['filters'] = self.filters
        if self.compressor:
            encoding['compressor'] = self.compressor
        data = self.data
        if self.dtype.kind == 'M':
            # xarray need a datetime64[ns] dtype
            data = data.astype('datetime64[ns]')
            encoding['dtype'] = 'int64'
        elif self.dtype.kind == 'm':
            encoding['dtype'] = 'int64'
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        if self.fill_value is not None:
            attrs['_FillValue'] = self.fill_value
        return xarray.Variable(self.dimensions, data, attrs, encoding)

    def __str__(self) -> str:
        return _variable_repr(self)

    def __repr__(self) -> str:
        return _variable_repr(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __array__(self):
        return self.values
