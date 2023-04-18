# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Delayed access to the chunked array.
====================================
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence
import uuid

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
from .abc import ModifiedVariableError, Variable, from_zarr_array


def _not_equal(first: Any, second: Any) -> bool:
    """Check if two objects are not equal.

    Args:
        first: The first object.
        second: The second object.

    Returns:
        True if the two objects are different, False otherwise.
    """

    def _is_not_a_number(number: Any) -> bool:
        """Check if a number is NaN or NaT."""
        # pylint: disable=comparison-with-itself
        return not number == number and number != number
        # pylint: enable=comparison-with-itself

    #: pylint: disable=unidiomatic-typecheck
    if type(first) != type(second):
        return True
    #: pylint: enable=unidiomatic-typecheck
    if _is_not_a_number(first) and _is_not_a_number(second):
        return False
    if first is None and second is None:
        return False
    if first == second:
        return False
    return True


def _asarray(
    arr: ArrayLike[Any],
    fill_value: Any | None = None,
) -> tuple[dask.array.core.Array, Any]:
    """Convert an array-like object to a dask array.

    Args:
        arr: An array-like object.
        fill_value: The fill value.

    Returns:
        If the data provided is a masked array, the functions return the array
        with masked data replaced by its fill value and the fill value of the
        offered masked array. Otherwise, the provided array and fill value.
    """
    result = dask.array.core.asarray(arr)  # type: dask.array.core.Array
    _meta = result._meta  # pylint: disable=protected-access
    if isinstance(_meta, numpy.ma.MaskedArray):
        if fill_value is not None and _not_equal(fill_value, _meta.fill_value):
            raise ValueError(
                f'The fill value {fill_value!r} does not match the fill value '
                f'{_meta.fill_value!r} of the array.')
        return dask.array.ma.filled(result, _meta.fill_value), _meta.fill_value
    return result, fill_value


def _new_variable(
    name: str,
    data: dask.array.core.Array,
    dimensions: Sequence[str],
    attrs: Sequence[Attribute],
    compressor: numcodecs.abc.Codec | None,
    fill_value: Any | None,
    filters: Sequence[numcodecs.abc.Codec] | None,
) -> DelayedArray:
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
    self = DelayedArray.__new__(DelayedArray)
    self.tableau = data
    self.attrs = attrs
    self.compressor = compressor
    self.dimensions = dimensions
    self.fill_value = fill_value
    self.filters = filters
    self.name = name
    return self


class DelayedArray(Variable):
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
    def TABLEAU(self) -> dask.array.core.Array:
        """Variable data as a dask array."""
        return self.tableau

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
        # If the fill value is None, or if the dask array already holds a
        # masked array, return the underlying array.
        # pylint: disable=protected-access
        # No other way to check if the dask array is a masked array.
        if (self.fill_value is None
                or isinstance(self.tableau._meta, numpy.ma.MaskedArray)):
            return self.tableau
        # pylint: enable=protected-access
        return dask.array.ma.masked_equal(self.tableau, self.fill_value)

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
        self.tableau, self.fill_value = data, fill_value

    @property
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """
        return self.compute()

    @property
    def dtype(self) -> numpy.dtype:
        """Return the dtype of the underlying array."""
        return self.tableau.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""
        return self.tableau.shape

    def persist(self, **kwargs) -> DelayedArray:
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        self.tableau = dask.base.persist(self.tableau,
                                         **kwargs)  # type: ignore
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
        try:
            (values, ) = dask.base.compute(self.tableau,
                                           traverse=False,
                                           **kwargs)
        except ValueError as exc:
            msg = str(exc)
            if 'cannot reshape' in msg or 'buffer too small' in msg:
                raise ModifiedVariableError() from exc
            raise
        return values if self.fill_value is None else numpy.ma.masked_equal(
            values, self.fill_value)

    def fill(self) -> DelayedArray:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            self.tableau = dask.array.creation.full_like(
                self.tableau, self.fill_value)
        return self

    def duplicate(self, data: Any) -> DelayedArray:
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
        result = DelayedArray(self.name, data, self.dimensions, self.attrs,
                              self.compressor, self.fill_value, self.filters)
        if len(result.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        return result

    @classmethod
    def from_zarr(cls, array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> DelayedArray:
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
        data = from_zarr_array(
            array,
            array.shape,
            array.chunks,
            name=f'{name}-{uuid.uuid1()}',
            **kwargs,
        )
        return _new_variable(name, data, array.attrs[dimension], attrs,
                             array.compressor, array.fill_value, array.filters)

    def concat(self, other: DelayedArray | Sequence[DelayedArray],
               dim: str) -> DelayedArray:
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
            return _new_variable(
                self.name,
                dask.array.core.concatenate(
                    (self.tableau, *(item.tableau for item in other)),
                    axis=axis),
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
            return _new_variable(self.name, self.tableau, self.dimensions,
                                 self.attrs, self.compressor, self.fill_value,
                                 self.filters)

    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.
        Returns:
            The variable slice.
        """
        return self.data[key]

    def isel(self, key: tuple[slice, ...]) -> DelayedArray:
        """Return a new variable with data selected along the given dimension
        indices.

        Args:
            key: Dimension indices to select

        Returns:
            The new variable
        """
        return _new_variable(self.name, self.tableau[key], self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def set_for_insertion(self) -> DelayedArray:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """
        return _new_variable(self.name, self.tableau, self.dimensions, tuple(),
                             self.compressor, self.fill_value, self.filters)

    def rename(self, name: str) -> DelayedArray:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """
        return _new_variable(name, self.tableau, self.dimensions, self.attrs,
                             self.compressor, self.fill_value, self.filters)

    def rechunk(self, **kwargs) -> DelayedArray:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """
        return _new_variable(self.name, self.tableau.rechunk(**kwargs),
                             self.dimensions, self.attrs, self.compressor,
                             self.fill_value, self.filters)

    def to_dask_array(self):
        """Return the underlying dask array.

        Returns:
            The underlying dask array

        .. seealso::

            :func:`dask.array.asarray`
        """
        return self.data

    def __dask_graph__(self) -> Mapping | None:
        """Return the dask Graph."""
        return self.data.__dask_graph__()

    def __dask_keys__(self) -> list:
        """Return the output keys for the Dask graph."""
        return self.data.__dask_keys__()

    def __dask_layers__(self) -> tuple:
        """Return the layers for the Dask graph."""
        return self.data.__dask_layers__()

    def __dask_tokenize__(self):
        """Return the token for the Dask graph."""
        return dask.base.normalize_token(
            (type(self), self.name, self.data, self.dimensions, self.attrs,
             self.fill_value))

    @staticmethod
    def __dask_optimize__(dsk: MutableMapping, keys: list,
                          **kwargs) -> MutableMapping:
        """Returns whether the Dask graph can be optimized.

        .. seealso::
            :meth:`dask.array.Array.__dask_optimize__`
        """
        return dask.array.core.Array.__dask_optimize__(dsk, keys, **kwargs)

    #: The default scheduler get to use for this object.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def _dask_finalize(self, results, array_func, *args,
                       **kwargs) -> DelayedArray:
        """Finalize the computation of the variable."""
        array = array_func(results, *args, **kwargs)
        if not isinstance(array, dask.array.core.Array):
            array = dask.array.core.from_array(array)
        return _new_variable(self.name, array, self.dimensions, self.attrs,
                             self.compressor, self.fill_value, self.filters)

    def __dask_postcompute__(self) -> tuple:
        """Return the finalizer and extra arguments to convert the computed
        results into their in-memory representation."""
        array_func, array_args = self.data.__dask_postcompute__()
        return self._dask_finalize, (array_func, ) + array_args

    def __dask_postpersist__(self) -> tuple:
        """Return the rebuilder and extra arguments to rebuild an equivalent
        Dask collection from a persisted or rebuilt graph."""
        array_func, array_args = self.data.__dask_postpersist__()
        return self._dask_finalize, (array_func, ) + array_args
