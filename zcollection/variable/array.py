# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Direct access to the chunked array.
===================================
"""
from __future__ import annotations

from typing import Any, Sequence

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.base
import dask.threaded
import numcodecs.abc
import numpy
import zarr

from ..meta import Attribute
from ..type_hints import ArrayLike, NDArray, NDMaskedArray
from .abc import Variable, not_equal


def _asarray(
    arr: ArrayLike[Any],
    fill_value: Any | None = None,
) -> tuple[NDArray, Any]:
    """Convert an array-like object to a numpy array.

    Args:
        arr: An array-like object.
        fill_value: The fill value.

    Returns:
        If the data provided is a masked array, the functions return the array
        with masked data replaced by its fill value and the fill value of the
        offered masked array. Otherwise, the provided array and fill value.
    """
    result = numpy.asanyarray(arr)
    if isinstance(result, numpy.ma.MaskedArray):
        if fill_value is not None and not_equal(fill_value, result.fill_value):
            raise ValueError(
                f'The fill value {fill_value!r} does not match the fill value '
                f'{result.fill_value!r} of the array.')
        return numpy.ma.filled(result, result.fill_value), result.fill_value
    return result, fill_value


def new_array(
    name: str,
    data: NDArray,
    dimensions: Sequence[str],
    attrs: Sequence[Attribute],
    compressor: numcodecs.abc.Codec | None,
    fill_value: Any | None,
    filters: Sequence[numcodecs.abc.Codec] | None,
) -> Array:
    """Create a new variable.

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """
    self = Array.__new__(Array)
    self.array = data
    self.attrs = attrs
    self.compressor = compressor
    self.dimensions = dimensions
    self.fill_value = fill_value
    self.filters = filters
    self.name = name
    return self


class Array(Variable):
    """Access to the chunked data using Dask arrays.

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """

    def __init__(self,
                 name: str,
                 data: ArrayLike[Any],
                 dimensions: Sequence[str],
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        array, fill_value = _asarray(data, fill_value)
        super().__init__(
            name,
            array,
            dimensions,
            attrs=attrs,
            compressor=compressor,
            fill_value=fill_value,
            filters=filters,
        )

    @property
    def data(self) -> dask.array.core.Array:
        """Return the underlying dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """
        if self.fill_value is None:
            return dask.array.core.from_array(self.array)
        return dask.array.ma.masked_equal(self.array, self.fill_value)

    @data.setter
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
        data, fill_value = _asarray(data, self.fill_value)
        if len(data.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        self.array, self.fill_value = data, fill_value

    @property
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """
        return self.array if self.fill_value is None else numpy.ma.masked_equal(
            self.array, self.fill_value)

    @property
    def dtype(self) -> numpy.dtype:
        """Return the dtype of the underlying array."""
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""
        return self.array.shape

    def persist(self, **kwargs) -> Array:
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        return self

    def compute(self, **kwargs) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.compute`.
        """
        return self.values

    def fill(self) -> Array:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            self.array = numpy.full_like(self.array, self.fill_value)
        return self

    def duplicate(self, data: Any) -> Array:
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
        result = Array(self.name, data, self.dimensions, self.attrs,
                       self.compressor, self.fill_value, self.filters)
        if len(result.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        return result

    @classmethod
    def from_zarr(cls, array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> Array:
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
        attrs = tuple(
            Attribute(k, v) for k, v in array.attrs.items() if k != dimension)
        return new_array(name, array[...], array.attrs[dimension], attrs,
                         array.compressor, array.fill_value, array.filters)

    def concat(self, other: Array | Sequence[Array], dim: str) -> Array:
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
        if not isinstance(other, Sequence):
            other = [other]
        if not other:
            raise ValueError('other must be a non-empty sequence')
        try:
            axis = self.dimensions.index(dim)
            return new_array(
                self.name,
                numpy.concatenate(
                    (self.array, *(item.array for item in other)), axis=axis),
                self.dimensions,
                self.attrs,
                self.compressor,
                self.fill_value,
                self.filters,
            )
        except ValueError:
            # If the concatenation dimension is not within the dimensions of the
            # variable, then the original variable is returned (i.e.
            # concatenation is not necessary).
            return new_array(self.name, self.array, self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.
        Returns:
            The variable slice.
        """
        return self.array[key]

    def isel(self, key: tuple[slice, ...]) -> Array:
        """Return a new variable with data selected along the given dimension
        indices.

        Args:
            key: Dimension indices to select

        Returns:
            The new variable
        """
        return new_array(self.name, self.array[key], self.dimensions,
                         self.attrs, self.compressor, self.fill_value,
                         self.filters)

    def set_for_insertion(self) -> Array:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """
        return new_array(self.name, self.array, self.dimensions, tuple(),
                         self.compressor, self.fill_value, self.filters)

    def rename(self, name: str) -> Array:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """
        return new_array(name, self.array, self.dimensions, self.attrs,
                         self.compressor, self.fill_value, self.filters)

    def rechunk(self, **kwargs) -> Array:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """
        return self

    def to_dask_array(self):
        """Return the underlying dask array.

        Returns:
            The underlying dask array

        .. seealso::

            :func:`dask.array.asarray`
        """
        return self.data
