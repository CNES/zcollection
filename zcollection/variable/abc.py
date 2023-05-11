# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset variable.
=================
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence, TypeVar
import abc
import collections

import dask.array.core
import numcodecs.abc
import numpy
import xarray
import zarr

from .. import mathematics, meta, representation
from ..meta import Attribute
from ..type_hints import ArrayLike, NDArray, NDMaskedArray

#: Generic type for a variable interface.
I = TypeVar('I', bound='IVariable')

#: Generic type for a dataset variable.
T = TypeVar('T', bound='Variable')


def new_variable(cls: type[I], **kwargs: Any) -> I:
    """Create a new variable.

    Args:
        constructor: Variable constructor.
        kwargs: Keyword arguments passed to the constructor.
    """
    self: I = cls.__new__(cls)
    # pylint: disable=expression-not-assigned
    # We use the set notation to evaluate the generator
    {
        setattr(self, key, value)  # type: ignore[func-returns-value]
        for key, value in kwargs.items()
    }
    # pylint: enable=expression-not-assigned
    return self


def concat(
    self: T,
    other: T | Sequence[T],
    concatenate: Callable,
    dim: str,
) -> T:
    """Concatenate a variable with another variable or a sequence of variables.

    Args:
        self: A variable.
        other: A variable or a sequence of variables.
        constructor: Variable constructor.
        concatenate: Function used to concatenate the arrays.
        dim: Dimension along which the arrays are concatenated.

    Returns:
        A new variable.

    Raises:
        ValueError: If ``other`` is empty or is not a sequence of ``self`` type.
    """
    if not isinstance(other, Sequence):
        other = [other]
    if not other:
        raise ValueError('other must be a non-empty sequence')

    # Self and other must have the same type
    if not all(isinstance(item, type(self)) for item in other):
        raise ValueError(f'other must be a sequence of {type(self)}')

    try:
        axis: int = self.dimensions.index(dim)
        return new_variable(
            type(self),
            name=self.name,
            array=concatenate((self.array, *(item.array for item in other)),
                              axis=axis),
            dimensions=self.dimensions,
            attrs=self.attrs,
            compressor=self.compressor,
            fill_value=self.fill_value,
            filters=self.filters,
        )
    except ValueError:
        # If the concatenation dimension is not within the dimensions of the
        # variable, then the original variable is returned (i.e.
        # concatenation is not necessary).
        return new_variable(type(self),
                            name=self.name,
                            array=self.array,
                            dimensions=self.dimensions,
                            attrs=self.attrs,
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)


def not_equal(first: Any, second: Any) -> bool:
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


def _variable_repr(var: Variable) -> str:
    """Get the string representation of a variable.

    Args:
        var: A variable.

    Returns:
        The string representation of the variable.
    """
    # Dimensions
    dims_str: str = representation.dimensions(
        dict(zip(var.dimensions, var.shape)))
    lines: list[str] = [
        f'<{var.__module__}.{var.__class__.__name__} {dims_str}>',
        f'{var.array!r}'
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


class IVariable(abc.ABC):
    """Define the interface for a variable."""

    @property
    @abc.abstractmethod
    def data(self) -> dask.array.core.Array:
        """Return the values as a dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """

    @property
    @abc.abstractmethod
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable values.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """

    @values.setter
    @abc.abstractmethod
    def values(self, data: Any) -> None:
        """Defines the values array. If the data provided is a masked array,
        it's converted to an array, where the masked values are replaced by the
        fill value of this instance.

        Args:
            data: The new data to use

        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """

    @abc.abstractmethod
    def persist(self: I, **kwargs) -> I:
        """Persist the variable data into memory.

        Args:
            **kwargs: Additional arguments to pass to the persist method.

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
            **kwargs: Additional arguments to pass to the compute method.
        """

    @abc.abstractmethod
    def fill(self: I) -> I:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """

    @classmethod
    @abc.abstractmethod
    def from_zarr(cls: type[I], array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> I:
        """Create a new variable from a zarr array.

        Args:
            array: The zarr array
            name: Name of the variable
            dimension: Name of the attribute that defines the dimensions of the
                variable
            **kwargs: Additional arguments to pass to the method.

        Returns:
            The variable
        """

    @abc.abstractmethod
    def concat(self: I, other: I | Sequence[I], dim: str) -> I:
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
    def rechunk(self: I, **kwargs) -> I:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """


class Variable(IVariable):
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
                 data: Any,
                 dimensions: Sequence[str],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        #: Variable name
        self.name: str = name
        #: Variable data as a dask array.
        self.array: Any = data
        #: Variable dimensions
        self.dimensions = tuple(dimensions)
        #: Variable attributes
        self.attrs = tuple(attrs or ())
        #: Compressor used to compress the data during writing data to disk
        self.compressor: numcodecs.abc.Codec | None = compressor
        #: Value to use for uninitialized values
        self.fill_value: Any | None = fill_value
        #: Filters to apply before writing data to disk
        self.filters = tuple(filters or ())

    @property
    def dtype(self) -> numpy.dtype:
        """Return the dtype of the underlying array."""
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the variable."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the variable."""
        return len(self.dimensions)

    @property
    def size(self: Any) -> int:
        """Return the size of the variable."""
        return mathematics.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Return the number of bytes used by the variable."""
        return self.size * self.dtype.itemsize

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
        result: T = type(self)(self.name,
                               data,
                               self.dimensions,
                               attrs=self.attrs,
                               compressor=self.compressor,
                               fill_value=self.fill_value,
                               filters=self.filters)
        if len(result.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        return result

    def isel(self: T, key: tuple[slice, ...]) -> T:
        """Return a new variable with data selected along the given dimension
        indices.

        Args:
            key: Dimension indices to select

        Returns:
            The new variable
        """
        return new_variable(type(self),
                            name=self.name,
                            array=self.array[key],
                            dimensions=self.dimensions,
                            attrs=self.attrs,
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)

    def set_for_insertion(self: T) -> T:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """
        return new_variable(type(self),
                            name=self.name,
                            array=self.array,
                            dimensions=self.dimensions,
                            attrs=tuple(),
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)

    def rename(self: T, name: str) -> T:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """
        return new_variable(type(self),
                            name=name,
                            array=self.array,
                            dimensions=self.dimensions,
                            attrs=self.attrs,
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)

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
        return meta.Variable(self.name,
                             self.dtype,
                             dimensions=self.dimensions,
                             attrs=self.attrs,
                             compressor=self.compressor,
                             fill_value=self.fill_value,
                             filters=self.filters)

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
        encoding: dict[str, Any] = {}
        if self.filters:
            encoding['filters'] = self.filters
        if self.compressor:
            encoding['compressor'] = self.compressor
        data: ArrayLike = self.array
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

    def __array__(self) -> ArrayLike:
        return self.values
