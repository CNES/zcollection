# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset variable.
=================
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence
import collections
import uuid

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.array.routines
import dask.array.wrap
import dask.base
import dask.threaded
import numcodecs.abc
import numpy
import xarray
import zarr

from . import mathematics, meta
from .meta import Attribute
from .type_hints import ArrayLike, NDArray, NDMaskedArray

#: The dask array getter used to access the data.
GETTER = dask.array.core.getter


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


def _dimensions_repr(dimensions: dict[str, int]) -> str:
    """Get the string representation of the dimensions.

    Args:
        dimensions: The dimensions.

    Returns:
        The string representation of the dimensions.
    """
    return str(tuple(f'{name}: {value}' for name, value in dimensions.items()))


def _calculate_column_width(items: Iterable) -> int:
    """Calculate the maximum width of a column.

    Args:
        items: An iterable of items.

    Returns:
        The maximum width of a column.
    """
    max_name = max(len(str(name)) for name in items)
    return max(max_name, 7)


def _maybe_truncate(obj: Any, max_size: int) -> str:
    """Truncate the string representation of an object to the given length.

    Args:
        obj: An object.
        max_size: The maximum length of the string representation.

    Returns:
        The string representation of the object.
    """
    result = str(obj)
    if len(result) > max_size:
        return result[:max_size - 3] + '...'
    return result


def _pretty_print(obj: Any, num_characters: int = 120) -> str:
    """Pretty print the object.

    Args:
        obj: An object.
        num_characters: The maximum number of characters per line.

    Returns:
        The pretty printed string representation of the object.
    """
    result = _maybe_truncate(obj, num_characters)
    return result + ' ' * max(num_characters - len(result), 0)


def _attributes_repr(attrs: Sequence[Attribute]) -> Iterator[str]:
    """Get the string representation of the attributes.

    Args:
        attrs: The attributes.

    Returns:
        The string representation of the attributes.
    """
    width = _calculate_column_width(item.name for item in attrs)
    for attr in attrs:
        name_str = f'    {attr.name:<{width}s}'
        yield _pretty_print(f'{name_str}: {attr.value!r}')


def _variable_repr(var: Variable) -> str:
    """Get the string representation of a variable.

    Args:
        var: A variable.

    Returns:
        The string representation of the variable.
    """
    # Dimensions
    dims_str = _dimensions_repr(dict(zip(var.dimensions, var.shape)))
    lines = [
        f'<{var.__module__}.{var.__class__.__name__} {dims_str}>',
        f'{var.array!r}'
    ]
    # Attributes
    if len(var.attrs):
        lines.append('  Attributes:')
        lines += _attributes_repr(var.attrs)
    # Filters
    if var.filters:
        lines.append('  Filters:')
        lines += [f'    {item!r}' for item in var.filters]
    # Compressor
    if var.compressor:
        lines.append('  Compressor:')
        lines += [f'    {var.compressor!r}']
    return '\n'.join(lines)


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
) -> Variable:
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
    self = Variable.__new__(Variable)
    self.array = data
    self.attrs = attrs
    self.compressor = compressor
    self.dimensions = dimensions
    self.fill_value = fill_value
    self.filters = filters
    self.name = name
    return self


def _blockdims_from_blockshape(
        shape: tuple[int, ...],
        chunks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Convert a blockshape to a blockdims tuple."""
    return tuple(((chunk_item, ) * (shape_item // chunk_item) +
                  ((shape_item % chunk_item, ) if shape_item %
                   chunk_item else ()) if shape_item else (0, ))
                 for shape_item, chunk_item in zip(shape, chunks))


def _from_zarr_array(
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
    __slots__ = ('array', 'attrs', 'compressor', 'dimensions', 'fill_value',
                 'filters', 'name')

    def __init__(self,
                 name: str,
                 data: ArrayLike[Any],
                 dimensions: Sequence[str],
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        array, fill_value = _asarray(data, fill_value)
        #: Variable name
        self.name = name
        #: Variable data as a dask array.
        self.array: dask.array.core.Array = array
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
    def dtype(self) -> numpy.dtype:
        """Return the data type of the variable."""
        return self.array.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the variable."""
        return len(self.dimensions)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""
        return self.array.shape

    @property
    def size(self: Any) -> int:
        """Return the size of the variable."""
        return mathematics.prod(self.shape)

    @property
    def nbytes(self):
        """Return the number of bytes used by the variable."""
        return self.size * self.dtype.itemsize

    def metadata(self) -> meta.Variable:
        """Get the variable metadata.

        Returns:
            Variable metadata
        """
        return meta.Variable(self.name, self.dtype, self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def persist(self, **kwargs) -> Variable:
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        self.array = dask.base.persist(self.data, **kwargs)  # type: ignore
        return self

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
                or isinstance(self.array._meta, numpy.ma.MaskedArray)):
            return self.array
        # pylint: enable=protected-access
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
        return self.compute()

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
            (values, ) = dask.base.compute(self.array,
                                           traverse=False,
                                           **kwargs)
        except ValueError as exc:
            msg = str(exc)
            if 'cannot reshape' in msg or 'buffer too small' in msg:
                raise ModifiedVariableError() from exc
            raise
        return values if self.fill_value is None else numpy.ma.masked_equal(
            values, self.fill_value)

    def fill(self) -> Variable:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            self.array = dask.array.creation.full_like(self.array,
                                                       self.fill_value)
        return self

    def isel(self, key: tuple[slice, ...]) -> Variable:
        """Return a new variable with data selected along the given dimension
        indices.

        Args:
            key: Dimension indices to select

        Returns:
            The new variable
        """
        return _new_variable(self.name, self.array[key], self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    @classmethod
    def from_zarr(cls, array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> Variable:
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
        data = _from_zarr_array(
            array,
            array.shape,
            array.chunks,
            name=f'{name}-{uuid.uuid1()}',
            **kwargs,
        )
        return _new_variable(name, data, array.attrs[dimension], attrs,
                             array.compressor, array.fill_value, array.filters)

    def duplicate(self, data: Any) -> Variable:
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
        result = Variable(self.name, data, self.dimensions, self.attrs,
                          self.compressor, self.fill_value, self.filters)
        if len(result.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        return result

    def set_for_insertion(self) -> Variable:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """
        return _new_variable(self.name, self.array, self.dimensions, tuple(),
                             self.compressor, self.fill_value, self.filters)

    def fill_attrs(self, var: meta.Variable):
        """Fill the variable attributes using the provided metadata.

        Args:
            var: Variable's metadata.
        """
        self.attrs = var.attrs

    def rename(self, name: str) -> Variable:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """
        return _new_variable(name, self.array, self.dimensions, self.attrs,
                             self.compressor, self.fill_value, self.filters)

    def dimension_index(self) -> Iterator[tuple[str, int]]:
        """Return an iterator over the variable dimensions and their index.

        Returns:
            An iterator over the variable dimensions
        """
        yield from ((item, ix) for ix, item in enumerate(self.dimensions))

    def concat(self, other: Variable | Sequence[Variable],
               dim: str) -> Variable:
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
            return _new_variable(self.name, self.array, self.dimensions,
                                 self.attrs, self.compressor, self.fill_value,
                                 self.filters)

    def rechunk(self, **kwargs) -> Variable:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """
        return _new_variable(self.name, self.array.rechunk(**kwargs),
                             self.dimensions, self.attrs, self.compressor,
                             self.fill_value, self.filters)

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
        data = self.array
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

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __array__(self):
        return self.values

    def to_dask_array(self):
        """Return the underlying dask array.

        Returns:
            The underlying dask array

        .. seealso::

            :func:`dask.array.asarray`
        """
        return self.array

    def __dask_graph__(self) -> Mapping | None:
        """Return the dask Graph."""
        return self.array.__dask_graph__()

    def __dask_keys__(self) -> list:
        """Return the output keys for the Dask graph."""
        return self.array.__dask_keys__()

    def __dask_layers__(self) -> tuple:
        """Return the layers for the Dask graph."""
        return self.array.__dask_layers__()

    def __dask_tokenize__(self):
        """Return the token for the Dask graph."""
        return dask.base.normalize_token(
            (type(self), self.name, self.array, self.dimensions, self.attrs,
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

    def _dask_finalize(self, results, array_func, *args, **kwargs):
        array = array_func(results, *args, **kwargs)
        if not isinstance(array, dask.array.core.Array):
            array = dask.array.core.from_array(array)
        return _new_variable(self.name, array, self.dimensions, self.attrs,
                             self.compressor, self.fill_value, self.filters)

    def __dask_postcompute__(self) -> tuple:
        """Return the finalizer and extra arguments to convert the computed
        results into their in-memory representation."""
        array_func, array_args = self.array.__dask_postcompute__()
        return self._dask_finalize, (array_func, ) + array_args

    def __dask_postpersist__(self) -> tuple:
        """Return the rebuilder and extra arguments to rebuild an equivalent
        Dask collection from a persisted or rebuilt graph."""
        array_func, array_args = self.array.__dask_postpersist__()
        return self._dask_finalize, (array_func, ) + array_args
