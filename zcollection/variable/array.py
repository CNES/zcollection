# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
In memory variable arrays.
==========================
"""
from __future__ import annotations

from typing import Any, Sequence

import dask.array.core
import dask.array.ma
import numcodecs.abc
import numpy
import zarr

from ..meta import Attribute
from ..type_hints import ArrayLike, NDArray, NDMaskedArray
from .abc import Variable, concat, new_variable, not_equal


def _as_numpy_array(
    arr: Any,
    *,
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
    result: NDArray = numpy.asanyarray(arr)
    if isinstance(result, numpy.ma.MaskedArray):
        if fill_value is not None and not_equal(fill_value, result.fill_value):
            raise ValueError(
                f'The fill value {fill_value!r} does not match the fill value '
                f'{result.fill_value!r} of the array.')
        return numpy.ma.filled(result, result.fill_value), result.fill_value
    return result, fill_value


class Array(Variable):
    """Access to the chunked data using numpy arrays.

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
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        array: NDArray
        array, fill_value = _as_numpy_array(data, fill_value=fill_value)
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
        """Return the numpy array wrapped in a dask array. If the variable has
        a fill value, the result is a masked array where masked values are
        equal to the fill value.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """
        if self.fill_value is None:
            return dask.array.core.from_array(self.array)
        return dask.array.ma.masked_equal(self.array, self.fill_value)

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

    @values.setter
    def values(self, data: Any) -> None:
        """Defines the underlying numpy array. If the data provided is a masked
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
        if len(data.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        self.array, self.fill_value = _as_numpy_array(
            data, fill_value=self.fill_value)

    def persist(self, **_) -> Array:
        """Persist the variable data into memory.

        Returns:
            The variable
        """
        return self

    def compute(self, **_) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.
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

    @classmethod
    def from_zarr(cls, array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> Array:
        """Create a new variable from a zarr array.

        Args:
            array: The zarr array
            name: Name of the variable
            dimension: Name of the attribute that defines the dimensions of the
                variable
            **kwargs: Additional arguments. These arguments are ignored, but
                they are accepted to be compatible with the base class.

        Returns:
            The variable
        """
        attrs = tuple(
            Attribute(k, v) for k, v in array.attrs.items() if k != dimension)
        return new_variable(cls,
                            name=name,
                            array=array[...],
                            dimensions=array.attrs[dimension],
                            attrs=attrs,
                            compressor=array.compressor,
                            fill_value=array.fill_value,
                            filters=tuple(array.filters or ()))

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
            ValueError: if the variables provided is an empty sequence or if
                any item in the sequence is not an instance of :class:`Array`.
        """
        return concat(self, other, numpy.concatenate, dim)

    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.
        Returns:
            The variable slice.
        """
        return self.array[key]

    def rechunk(self, **_) -> Array:
        """Rechunk the variable.

        Returns:
            The variable.
        """
        return self
