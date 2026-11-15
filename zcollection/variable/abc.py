# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset variable.
=================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeVar
import abc
import collections
from collections.abc import Callable, Iterator, Sequence
import numbers

import dask.array.core
import dask.array.ma
import numpy
import xarray

from .. import mathematics, meta, representation, serialization

if TYPE_CHECKING:
    import zarr

    from ..codecs import Codec
    from ..meta import Attribute
    from ..type_hints import ArrayLike, DTypeLike, NDArray, NDMaskedArray

__all__ = ('Variable', )

#: Name of the :data:`FILL_VALUE` attribute used to store a fill value.
#: This attribute is written only when the variable has a fill value.
FILL_VALUE = '_FillValue'

#: Name of the Zarr 2 attribute containing array dimension names.
DIMENSIONS = '_ARRAY_DIMENSIONS'

#: Zarr version that stores dimension names in an array attribute.
ZARR_FORMAT_2 = 2

#: Generic type for a variable interface.
I = TypeVar('I', bound='IVariable')

#: Generic type for a dataset variable.
T = TypeVar('T', bound='Variable')


def concat(
    self: T,
    other: T | Sequence[T],
    concatenate: Callable,
    dim: str,
) -> T:
    """Concatenate a variable with one or more variables.

    Args:
        self: The source variable.
        other: Another variable or a sequence of variables.
        concatenate: The function used to concatenate their arrays.
        dim: The concatenation dimension.

    Returns:
        The new variable. If ``dim`` is absent from its dimensions,
        a copy of the source variable is returned.

    Raises:
        ValueError: If ``other`` is empty or contains incompatible types.
    """
    if not isinstance(other, Sequence):
        other = [other]
    if not other:
        raise ValueError('other must be a non-empty sequence')

    # All variables must have the same type.
    if not all(isinstance(item, type(self)) for item in other):
        raise ValueError(f'other must be a sequence of {type(self)}')

    # The result is a filler if any input variable is a filler.
    filler: bool = any(item.filler for item in (self, *other))
    try:
        axis: int = self.dimensions.index(dim)
        return self.duplicate(concatenate(
            (self.array, *(item.array for item in other)), axis=axis),
                              filler=filler)
    except ValueError:
        # Return a copy when concatenation is not applicable to this dimension.
        return self.duplicate(filler=filler)


def read_attributes(array: Any) -> tuple[tuple[meta.Attribute, ...], Any]:
    """Read the attributes and fill value of a Zarr array.

    Args:
        array: The Zarr array.

    Returns:
        The variable attributes and its decoded fill value, or ``None`` if
        none is defined. The :data:`FILL_VALUE` attribute stores the encoded
        fill value, which is decoded using the array's data type. For Zarr 2
        arrays without this attribute, the array's fill value is used.
    """
    attrs = tuple(
        meta.Attribute(name, value) for name, value in array.attrs.items()
        if name not in (DIMENSIONS, FILL_VALUE))
    if FILL_VALUE in array.attrs:
        return attrs, serialization.decode_fill_value(array.attrs[FILL_VALUE],
                                                      array.dtype)
    if array.metadata.zarr_format == ZARR_FORMAT_2:
        return attrs, array.fill_value
    return attrs, None


def read_dimensions(array: Any) -> tuple[str, ...]:
    """Return the names of a Zarr array's dimensions.

    Args:
        array: The Zarr array.

    Returns:
        The dimension names. Zarr 2 arrays store them in the
        :data:`DIMENSIONS` attribute, while Zarr 3 arrays use the
        ``dimension_names`` metadata field.
    """
    if array.metadata.zarr_format == ZARR_FORMAT_2:
        return tuple(array.attrs.get(DIMENSIONS, ()))
    return tuple(array.metadata.dimension_names or ())


def read_compressor(array: Any) -> Any:
    """Read the compressor of a Zarr array.

    Args:
        array: The Zarr array.

    Returns:
        The first compressor codec, or ``None`` if the array has none.
    """
    compressors = array.compressors
    return compressors[0] if compressors else None


def _is_itself(value: Any) -> bool:
    """Return whether a value equals itself.

    Args:
        value: The value to compare.

    Returns:
        ``False`` for NaN or NaT; ``True`` for values equal to themselves.
    """
    return bool(value == value)  # noqa: PLR0124


def _differs_from_itself(values: Any) -> Any:
    """Identify array elements that are not equal to themselves.

    Args:
        values: The array to inspect.

    Returns:
        A boolean array that is ``True`` for NaN or NaT elements.
    """
    return values != values  # noqa: PLR0124


def masked(values: Any, fill_value: Any) -> Any:
    """Mask array elements matching a fill value.

    Args:
        values: The array to mask.
        fill_value: The value used to identify missing elements.

    Returns:
        The array with elements matching the fill value masked. When the
        fill value is NaN or NaT, matching elements are found by comparing
        each element with itself, since these values are not equal to
        themselves.
    """
    module = (dask.array
              if isinstance(values, dask.array.core.Array) else numpy)
    if _is_itself(fill_value):
        return module.ma.masked_equal(values, fill_value)
    return module.ma.masked_where(_differs_from_itself(values), values)


def not_equal(first: Any, second: Any) -> bool:
    """Compare two values, treating matching NaN and NaT values as equal.

    Args:
        first: The first value.
        second: The second value.

    Returns:
        Whether the values differ.
    """

    def _is_not_a_number(number: Any) -> bool:
        """Return whether a value is NaN or NaT."""

        return not number == number and number != number  # noqa: PLR0124

    if (not isinstance(first, numbers.Number) or not isinstance(
            second, numbers.Number)) and type(first) is not type(second):
        return True
    if _is_not_a_number(first) and _is_not_a_number(second):
        return False
    if first is None and second is None:
        return False
    return not (first == second)


def _variable_repr(var: Variable) -> str:
    """Format a variable as a string.

    Args:
        var: The variable.

    Returns:
        The formatted string.
    """
    # Dimensions
    dims_str: str = representation.dimensions(
        dict(zip(var.dimensions, var.shape, strict=True)))
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
        """Return variable data as a Dask array, masking fill values if defined.

        An eager variable is wrapped in a Dask array. The returned data is
        masked when a fill value is defined.

        Returns:
            The Dask array.

        .. seealso::

            :attr:`Variable.array <zcollection.variable.abc.Variable.array>`
        """

    @property
    @abc.abstractmethod
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable values as a NumPy array.

        Values matching a defined fill value are masked.

        Returns:
            The variable values, masked if a fill value is defined.
        """

    @values.setter
    @abc.abstractmethod
    def values(self, data: Any) -> None:
        """Replace the variable's data.

        For masked input, missing values are replaced by the variable's fill
        value. If the variable has none, the masked array's fill value is used
        and becomes the variable's fill value. Unmasked input preserves the
        existing fill value.

        Args:
            data: The new data.

        Raises:
            ValueError: If the number of data dimensions does not match the
                number of variable dimensions, or if the variable and the
                masked input have different fill values.
        """

    @abc.abstractmethod
    def persist(self: I, **kwargs) -> I:
        """Persist the variable data in memory.

        Args:
            **kwargs: Additional arguments passed to the persistence method.

        Returns:
            The variable.
        """

    @abc.abstractmethod
    def compute(self, **kwargs) -> NDArray | NDMaskedArray:
        """Compute the variable values as a NumPy array.

        Values matching a defined fill value are masked.

        Args:
            **kwargs: Additional arguments passed to the compute method.
        """

    @abc.abstractmethod
    def fill(self: I) -> I:
        """Fill the variable with its fill value, if one is defined.

        Returns:
            The variable, unchanged if no fill value is defined.
        """

    @classmethod
    @abc.abstractmethod
    def from_zarr(cls: type[I], array: zarr.Array, name: str, **kwargs) -> I:
        """Create a variable from a Zarr array.

        Args:
            array: The Zarr array.
            name: The variable name.
            **kwargs: Additional arguments passed to the method.

        Returns:
            The new variable.
        """

    @abc.abstractmethod
    def concat(self: I, other: I | Sequence[I], dim: str) -> I:
        """Concatenate the variable with other variables along a dimension.

        Args:
            other: A variable or sequence of variables to concatenate.
            dim: The dimension along which to concatenate.

        Returns:
            The concatenated variable.

        Raises:
            ValueError: If the sequence of variables is empty.
        """

    @abc.abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Return the values selected by an index or slice.

        Args:
            key: The index or slice.

        Returns:
            The selected values.
        """

    @abc.abstractmethod
    def rechunk(self: I, **kwargs) -> I:
        """Rechunk the variable.

        Args:
            **kwargs: Arguments passed to :func:`dask.array.rechunk`.

        Returns:
            The variable.
        """


class Variable(IVariable, abc.ABC):
    """Represent a multidimensional array with associated metadata.

    Args:
        name: The variable name.
        data: The variable data.
        dimensions: The names of the variable dimensions.
        attrs: The variable attributes.
        compressor: The compression codec.
        fill_value: The value used to mask missing elements when reading
            the variable. NaN and NaT elements are also masked when the
            fill value is NaN or NaT, respectively.
        filters: The filters applied before writing data to storage.
        filler: Whether the variable was generated by the system.
    """
    __slots__ = (
        'array',
        'attrs',
        'compressor',
        'dimensions',
        'fill_value',
        'filler',
        'filters',
        'name',
    )

    def __init__(self,
                 name: str,
                 data: Any,
                 dimensions: Sequence[str],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[Codec] | None = None,
                 filler: bool = False) -> None:
        #: Variable name
        self.name: str = name
        #: The underlying NumPy or Dask array.
        self.array: Any = data
        #: Variable dimensions
        self.dimensions = tuple(dimensions)
        #: Variable attributes
        self.attrs = tuple(attrs or ())
        #: Compressor used when writing the variable to storage.
        self.compressor: Codec | None = compressor
        #: Fill value used for uninitialized data and masking.
        self.fill_value: Any | None = fill_value
        #: Filter codecs applied before writing data.
        self.filters = tuple(filters or ())
        #: Origin of the variable
        self.filler = filler

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

    def duplicate(self, data: Any = None, **properties: Any) -> Self:
        """Create a new variable using this variable's properties.

        Args:
            data: The new data. Defaults to the current variable data.
            **properties: Properties to set on the new variable.

        Returns:
            The new variable.

        Raises:
            TypeError: If a property name is not supported.
            ValueError: If the number of data dimensions does not match the
                dimensions declared for the new variable.
        """
        kwargs: dict[str, Any] = {
            name: properties.pop(name, getattr(self, name))
            for name in self.__slots__ if name != 'array'
        }
        if properties:
            raise TypeError('A variable holds no property named '
                            f'{", ".join(sorted(properties))}')
        result: Self = type(self)(kwargs.pop('name'),
                                  self.array if data is None else data,
                                  kwargs.pop('dimensions'), **kwargs)
        if len(result.shape) != len(result.dimensions):
            raise ValueError('Data shape does not match variable dimensions')
        return result

    def isel(self, key: tuple[slice, ...]) -> Self:
        """Return a new variable with data selected by positional indices.

        Args:
            key: A tuple of indices applied to the variable's array.

        Returns:
            The selected variable.
        """
        return self.duplicate(self.array[key])

    def rename(self, name: str) -> Self:
        """Return a copy of the variable with a different name.

        Args:
            name: The new variable name.

        Returns:
            The renamed variable.
        """
        return self.duplicate(name=name)

    def fill_attrs(self, var: meta.VariableSchema) -> None:
        """Replace the variable attributes using a variable schema.

        Args:
            var: The variable schema providing the attributes.
        """
        self.attrs = var.attrs

    def metadata(self) -> meta.VariableSchema:
        """Return the variable metadata.

        Returns:
            The variable metadata.
        """
        return meta.VariableSchema(self.name,
                                   self.dtype,
                                   dimensions=self.dimensions,
                                   attrs=self.attrs,
                                   compressor=self.compressor,
                                   fill_value=self.fill_value,
                                   filters=self.filters)

    def have_same_properties(self, other: Variable) -> bool:
        """Return whether two variables have the same metadata properties."""
        return self.metadata() == other.metadata()

    def dimension_index(self) -> Iterator[tuple[str, int]]:
        """Iterate over dimension names and their indices in the variable.

        Returns:
            An iterator of dimension names and indices.
        """
        yield from ((item, ix) for ix, item in enumerate(self.dimensions))

    def to_xarray(self) -> xarray.Variable:
        """Convert the variable to an xarray.Variable.

        Returns:
            The xarray variable.
        """
        encoding: dict[str, Any] = {}
        if self.filters:
            encoding['filters'] = self.filters
        if self.compressor:
            encoding['compressor'] = self.compressor
        data: ArrayLike = self.array
        if self.dtype.kind == 'M':
            # xarray requires datetime values with ``datetime64[ns]`` precision.
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

    def __array__(self,
                  dtype: DTypeLike | None = None,
                  copy: bool | None = None) -> NDArray | NDMaskedArray:
        """Return the values with the requested NumPy dtype and copy policy.

        A conversion without copying requires unmasked NumPy data and a
        matching dtype. Other variables must compute or mask their values.
        """
        if copy is False and (not isinstance(self.array, numpy.ndarray)
                              or self.fill_value is not None):
            raise ValueError('This variable cannot be converted without '
                             'a copy')
        return numpy.array(self.values, dtype=dtype, copy=copy, subok=True)
