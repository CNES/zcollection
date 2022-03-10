# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset
=======
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import collections
import functools
import operator

import dask.array
import dask.base
import dask.threaded
import numcodecs.abc
import numpy
import xarray

from . import meta
from .meta import Attribute
from .typing import ArrayLike, NDArray, NDMaskedArray


def _prod(iterable: Iterable) -> int:
    """Get the product of an iterable.

    Args:
        iterable: An iterable.

    Returns:
        The product of the iterable.
    """
    return functools.reduce(operator.mul, iterable, 1)


def _dimensions_repr(dimensions: Dict[str, int]) -> str:
    """Get the string representation of the dimensions.

    Args:
        dimensions: The dimensions.

    Returns:
        The string representation of the dimensions.
    """
    return str(tuple(f"{name}: {value}" for name, value in dimensions.items()))


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
        return result[:max_size - 3] + "..."
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
    return result + " " * max(num_characters - len(result), 0)


def _dask_repr(array: dask.array.Array) -> str:
    """Get the string representation of a dask array.

    Args:
        array: A dask array.

    Returns:
        The string representation of the dask array.
    """
    chunksize = tuple(item[0] for item in array.chunks)
    return f"dask.array<chunksize={chunksize}>"


def _attributes_repr(attrs: Sequence[Attribute]) -> Iterator[str]:
    """Get the string representation of the attributes.

    Args:
        attrs: The attributes.

    Returns:
        The string representation of the attributes.
    """
    width = _calculate_column_width(item.name for item in attrs)
    for attr in attrs:
        name_str = f"    {attr.name:<{width}s}"
        yield _pretty_print(f"{name_str}: {attr.value!r}")


def _dataset_repr(ds: "Dataset") -> str:
    """Get the string representation of a dataset.

    Args:
        ds: A dataset.

    Returns:
        The string representation of the dataset.
    """
    # Dimensions
    dims_str = _dimensions_repr(ds.dimensions)
    lines = [
        f"<{ds.__module__}.{ds.__class__.__name__}>",
        f"  Dimensions: {dims_str}", "Data variables:"
    ]
    # Variables
    if len(ds.variables) == 0:
        lines.append("    <empty>")
    else:
        width = _calculate_column_width(ds.variables)
        for name, variable in ds.variables.items():
            dims_str = f"({', '.join(map(str, variable.dimensions))} "
            name_str = f"    {name:<{width}s} {dims_str} {variable.dtype}"
            lines.append(
                _pretty_print(f"{name_str}: {_dask_repr(variable.array)}"))
    # Attributes
    if len(ds.attrs):
        lines.append("  Attributes:")
        lines += _attributes_repr(ds.attrs)

    return "\n".join(lines)


def _variable_repr(var: "Variable") -> str:
    """Get the string representation of a variable

    Args:
        var: A variable.

    Returns:
        The string representation of the variable.
    """
    # Dimensions
    dims_str = _dimensions_repr(dict(zip(var.dimensions, var.shape)))
    lines = [
        f"<{var.__module__}.{var.__class__.__name__} {dims_str}>",
        f"{var.array!r}"
    ]
    # Attributes
    if len(var.attrs):
        lines.append("  Attributes:")
        lines += _attributes_repr(var.attrs)
    # Filters
    if var.filters:
        lines.append("  Filters:")
        lines += [f"    {item!r}" for item in var.filters]
    # Compressor
    if var.compressor:
        lines.append("  Compressor:")
        lines += [f"    {var.compressor!r}"]
    return "\n".join(lines)


def _asarray(
    arr: ArrayLike,
    fill_value: Optional[Any] = None,
) -> Tuple[dask.array.Array, Any]:
    """Convert an array-like object to a dask array.

    Args:
        arr: An array-like object.
        fill_value: The fill value.

    Returns:
        If the data provided is a masked array, the functions return the array
        with masked data replaced by its fill value and the fill value of the
        offered masked array. Otherwise, the provided array and fill value.
    """
    result = dask.array.asarray(arr)  # type: dask.array.Array
    _meta = result._meta  # pylint: disable=protected-access
    if isinstance(_meta, numpy.ma.MaskedArray):
        if fill_value is not None and fill_value != _meta.fill_value:
            raise ValueError(
                f"The fill value {fill_value!r} does not match the fill value "
                f"{_meta.fill_value!r} of the array.")
        return dask.array.ma.filled(result, _meta.fill_value), _meta.fill_value
    return result, fill_value


class Variable:
    """Variables hold multi-dimensional arrays of data

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """
    __slots__ = ("_array", "name", "dimensions", "attrs", "compressor",
                 "fill_value", "filters")

    def __init__(
            self,
            name: str,
            data: ArrayLike,
            dimensions: Sequence[str],
            attrs: Optional[Sequence[Attribute]] = None,
            compressor: Optional[numcodecs.abc.Codec] = None,
            fill_value: Optional[Any] = None,
            filters: Optional[Sequence[numcodecs.abc.Codec]] = None) -> None:
        array, fill_value = _asarray(data, fill_value)
        #: Variable name
        self.name = name
        #: Variable data as a dask array.
        self._array = array
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
        """Return the data type of the variable"""
        return self._array.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the variable.
        """
        return len(self.dimensions)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the variable"""
        return self._array.shape

    @property
    def size(self: Any) -> int:
        """Return the size of the variable"""
        return _prod(self.shape)

    @property
    def nbytes(self):
        """Return the number of bytes used by the variable"""
        return self.size * self.dtype.itemsize

    def metadata(self) -> meta.Variable:
        """Get the variable metadata

        Returns:
            Variable metadata
        """
        return meta.Variable(self.name, self.dtype, self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def have_same_properties(self, other: "Variable") -> bool:
        """Return true if this instance and the other variable have the same
        properties"""
        return self.metadata() == other.metadata()

    @property
    def array(self) -> dask.array.Array:
        """Return the dask array underlying the variable.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.data`
        """
        return self._array

    def persist(self, **kwargs) -> "Variable":
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        self._array = self._array.persist(**kwargs)
        return self

    @property
    def data(self) -> dask.array.Array:
        """Return the underlying dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """
        return (self._array if self.fill_value is None else
                dask.array.ma.masked_equal(self._array, self.fill_value))

    @data.setter
    def data(self, data: Any) -> None:
        """Defines the underlying dask array. If the data provided is a masked
        array, it's converted to an array, where the masked values are replaced
        by its fill value, and its fill value becomes the new fill value of this
        instance. Otherwise, the underlying array is defined as the new data
        and the fill value is set to None.

        Args:
            data: The new data to use

        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """
        data, fill_value = _asarray(data, self.fill_value)
        if len(data.shape) != len(self.dimensions):
            raise ValueError("data shape does not match variable dimensions")
        self._array, self.fill_value = data, fill_value

    @property
    def values(self) -> Union[NDArray, NDMaskedArray]:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """
        return self.data.compute()

    def duplicate(self, data: Any) -> "Variable":
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
        data = dask.array.asarray(data)
        if len(data.shape) != len(self.dimensions):
            raise ValueError("data shape does not match variable dimensions")
        return Variable(self.name, data, self.dimensions, self.attrs,
                        self.compressor, self.fill_value, self.filters)

    def dimension_index(self) -> Iterator[Tuple[str, int]]:
        """Return an iterator over the variable dimensions and their index

        Returns:
            An iterator over the variable dimensions
        """
        for ix, item in enumerate(self.dimensions):
            yield item, ix

    def concat(self, other: Union["Variable", Sequence["Variable"]],
               dim: str) -> "Variable":
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
            raise ValueError("other must be a non-empty sequence")
        try:
            axis = self.dimensions.index(dim)
            result = self.duplicate(self._array)
            # pylint: disable=protected-access
            # _array is a protected member of this class
            result._array = dask.array.concatenate(
                [self._array, *[item._array for item in other]], axis=axis)
            # pylint: enable=protected-access
            return result
        except ValueError:
            # If the concatenation dimension is not within the dimensions of the
            # variable, then the original variable is returned (i.e.
            # concatenation is not necessary).
            return self.duplicate(self._array)

    def to_xarray(self) -> xarray.Variable:
        """Convert the variable to an xarray.Variable

        Returns:
            Variable as an xarray.Variable
        """
        encoding = {}
        if self.filters:
            encoding["filters"] = self.filters
        if self.compressor:
            encoding["compressor"] = self.compressor
        data = self._array
        if self.dtype.kind == "M":
            # xarray need a datetime64[ns] dtype
            data = data.astype("datetime64[ns]")
            encoding["dtype"] = "int64"
        elif self.dtype.kind == "m":
            encoding["dtype"] = "int64"
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        if self.fill_value is not None:
            attrs["_FillValue"] = self.fill_value
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

            :func:`dask.array.core.asarray`
        """
        return self.data

    def __dask_graph__(self) -> Optional[Mapping]:
        """Return the dask Graph"""
        return self._array.__dask_graph__()

    def __dask_keys__(self) -> List:
        """Return the output keys for the Dask graph"""
        return self._array.__dask_keys__()

    def __dask_layers__(self) -> Tuple:
        """Return the layers for the Dask graph"""
        return self._array.__dask_layers__()

    def __dask_tokenize__(self):
        """Return the token for the Dask graph"""
        return dask.base.normalize_token(
            (type(self), self.name, self._array, self.dimensions, self.attrs,
             self.fill_value))

    @staticmethod
    def __dask_optimize__(dsk: MutableMapping, keys: List,
                          **kwargs) -> MutableMapping:
        """Returns whether the Dask graph can be optimized

        .. seealso::
            :func:`dask.array.Array.__dask_optimize__`
        """
        return dask.array.Array.__dask_optimize__(dsk, keys, **kwargs)

    #: The default scheduler get to use for this object.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def _dask_finalize(self, results, array_func, *args, **kwargs):
        data = array_func(results, *args, **kwargs)
        return Variable(self.name, data, self.dimensions, self.attrs,
                        self.compressor, self.fill_value, self.filters)

    def __dask_postcompute__(self) -> Tuple:
        """Return the finalizer and extra arguments to convert the computed
        results into their in-memory representation.
        """
        array_func, array_args = self._array.__dask_postcompute__()
        return self._dask_finalize, (array_func, ) + array_args

    def __dask_postpersist__(self) -> Tuple:
        """Return the rebuilder and extra arguments to rebuild an equivalent
        Dask collection from a persisted or rebuilt graph.
        """
        array_func, array_args = self._array.__dask_postpersist__()
        return self._dask_finalize, (array_func, ) + array_args


class Dataset:
    """Hold variables, dimensions, and attributes that together form a dataset.

    Attrs:
        variables: Dataset variables
        attrs: Dataset attributes

    Raises:
        ValueError: If the dataset contains variables with the same dimensions
            but with different values.
    """
    __slots__ = ("dimensions", "variables", "attrs")

    def __init__(self,
                 variables: Iterable[Variable],
                 attrs: Optional[Sequence[Attribute]] = None) -> None:
        #: The list of global attributes on this dataset
        self.attrs = attrs or []
        #: Dataset contents as dict of :py:class:`Variable` objects.
        self.variables = collections.OrderedDict(
            (item.name, item) for item in variables)
        #: A dictionary of dimension names and their index in the dataset
        self.dimensions: Dict[str, int] = {}

        for var in self.variables.values():
            for ix, dim in enumerate(var.dimensions):
                if dim not in self.dimensions:
                    self.dimensions[dim] = var.array.shape[ix]
                elif self.dimensions[dim] != var.array.shape[ix]:
                    raise ValueError(f"variable {var.name} has conflicting "
                                     "dimensions")

    def __getitem__(self, name: str) -> Variable:
        """Return a variable from the dataset.

        Args:
            name: The name of the variable to return

        Returns:
            The variable

        Raises:
            KeyError: If the variable is not found
        """
        return self.variables[name]

    def __getstate__(self) -> Tuple[Any, ...]:
        return self.dimensions, self.variables, self.attrs

    def __setstate__(self, state: Tuple[Any, ...]) -> None:
        self.dimensions, self.variables, self.attrs = state

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes in the dataset.

        Returns:
            The total number of bytes in the dataset
        """
        return sum(item.nbytes for item in self.variables.values())

    def add_variable(self,
                     variable: meta.Variable,
                     /,
                     data: Optional[ArrayLike] = None):
        """Add a variable to the dataset.

        Args:
            variable: The variable to add
            data: The data to add to the variable. If not provided, the variable
                will be created with the default fill value.

        Raises:
            ValueError: If the variable added has dimensions that conflict with
                existing dimensions, or if the variable has dimensions not
                defined in the dataset.
        """
        if set(variable.dimensions) - set(self.dimensions):
            raise ValueError(
                f"variable {variable.name} has dimensions "
                f"{variable.dimensions} that are not in the dataset")

        if data is None:
            shape = tuple(self.dimensions[dim] for dim in variable.dimensions)
            data = dask.array.full(shape,
                                   variable.fill_value,
                                   dtype=variable.dtype)
        else:
            for dim, size in zip(variable.dimensions,
                                 data.shape):  # type: ignore
                if size != self.dimensions[dim]:
                    raise ValueError(
                        f"Conflicting sizes for dimension {dim!r}: "
                        f"length {self.dimensions[dim]} on the data but length "
                        f"{size} defined in dataset.")
        self.variables[variable.name] = Variable(
            variable.name,
            data,  # type: ignore
            variable.dimensions,
            variable.attrs,
            variable.compressor,
            variable.fill_value,
            variable.filters,
        )

    def drops_vars(self, names: Union[str, Sequence[str]]) -> None:
        """Drop variables from the dataset.

        Args:
            names: Variable names to drop.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        # pylint: disable=expression-not-assigned
        {self.variables.pop(name) for name in names}
        # pylint: enable=expression-not-assigned

    def metadata(self) -> meta.Dataset:
        """Get the dataset metadata

        Returns:
            Dataset metadata
        """
        return meta.Dataset(dimensions=tuple(self.dimensions.keys()),
                            variables=tuple(
                                item.metadata()
                                for item in self.variables.values()),
                            attrs=self.attrs)

    @staticmethod
    def from_xarray(ds: xarray.Dataset) -> "Dataset":
        """Create a new dataset from an xarray dataset.

        Args:
            ds: Dataset to convert.

        Returns:
            New dataset.
        """
        variables = [
            Variable(
                name,  # type: ignore
                array.data,
                tuple(array.dims),
                tuple(
                    Attribute(*attr)  # type: ignore
                    for attr in array.attrs.items()),
                array.encoding.get("compressor", None),
                array.encoding.get("_FillValue", None),
                array.encoding.get("filters", None))
            for name, array in ds.variables.items()
        ]

        return Dataset(
            variables=variables,
            attrs=tuple(
                Attribute(*item)  # type: ignore
                for item in ds.attrs.items()))

    def to_xarray(self, **kwargs) -> xarray.Dataset:
        """Convert the dataset to an xarray dataset.

        Args:
            **kwargs: Additional parameters are passed through the function
                :py:func:`xarray.conventions.decode_cf_variables`.

        Returns:
            Dataset as an xarray dataset.
        """
        data_vars = collections.OrderedDict(
            (name, variable.to_xarray())
            for name, variable in self.variables.items())
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        data_vars, attrs, coord_names = xarray.conventions.decode_cf_variables(
            data_vars, attrs, **kwargs)
        ds = xarray.Dataset(data_vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(data_vars))
        return ds

    def isel(self, slices: Dict[str, Any]) -> "Dataset":
        """Return a new dataset with each array indexed along the specified
        slices.

        Args:
            slices: Dictionary of dimension names and slices

        Returns:
            New dataset.
        """
        dims_invalid = set(slices) - set(self.dimensions)
        if dims_invalid:
            raise ValueError(
                f"Slices contain invalid dimension name(s): {dims_invalid}")
        variables = [
            var.duplicate(var.array[tuple(
                slices[dim] if dim in slices else slice(None)
                for dim in var.dimensions)])
            for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def delete(self, indexer: Union[slice, Sequence[int]],
               axis: str) -> "Dataset":
        """Return a new dataset without the data selected by the provided
        indices.

        Args:
            indexer: Indices to remove along the specified axis.
            axis: The axis along which to delete the subarrays defined
                in the dataset.

        Returns:
            New dataset.
        """
        variables = [
            var.duplicate(
                dask.array.delete(var.array, indexer,
                                  var.dimensions.index(axis)))
            for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def compute(self) -> "Dataset":
        """Compute the dataset variables.

        Returns:
            New dataset.
        """
        variables = [
            var.duplicate(var.values) for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def persist(self, **kwargs) -> "Dataset":
        """Persist the dataset variables.

        Args:
            **kwargs: Additional parameters are passed to the function
                :py:func:`dask.array.Array.persist`.

        Returns:
            The dataset with the variables persisted into memory.
        """
        for variable in self.variables.values():
            variable.persist(**kwargs)
        return self

    def concat(self, other: Union["Dataset", Iterable["Dataset"]],
               dim: str) -> "Dataset":
        """Concatenate datasets along a dimension.

        Args:
            other: Datasets to concatenate.
            dim: Dimension along which to concatenate the datasets.

        Returns:
            New dataset.

        Raises:
            ValueError: If the provided sequence of datasets is empty.
        """
        variables = []
        if not isinstance(other, Iterable):
            other = [other]
        if not other:
            raise ValueError("cannot concatenate an empty sequence")
        variables = [
            variable.concat(tuple(item.variables[name] for item in other), dim)
            for name, variable in self.variables.items()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)
