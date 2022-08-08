# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset
=======
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import collections

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.array.routines
import dask.array.wrap
import dask.base
import dask.threaded
import xarray

from . import meta
from .meta import Attribute
from .typing import ArrayLike, NDArray, NDMaskedArray
from .variable import (
    Variable,
    _attributes_repr,
    _calculate_column_width,
    _dimensions_repr,
    _pretty_print,
)


def _dask_repr(array: dask.array.core.Array) -> str:
    """Get the string representation of a dask array.

    Args:
        array: A dask array.

    Returns:
        The string representation of the dask array.
    """
    chunksize = tuple(item[0] for item in array.chunks)
    return f'dask.array<chunksize={chunksize}>'


def _dataset_repr(ds: Dataset) -> str:
    """Get the string representation of a dataset.

    Args:
        ds: A dataset.

    Returns:
        The string representation of the dataset.
    """
    # Dimensions
    dims_str = _dimensions_repr(ds.dimensions)
    lines = [
        f'<{ds.__module__}.{ds.__class__.__name__}>',
        f'  Dimensions: {dims_str}', 'Data variables:'
    ]
    # Variables
    if len(ds.variables) == 0:
        lines.append('    <empty>')
    else:
        width = _calculate_column_width(ds.variables)
        for name, variable in ds.variables.items():
            dims_str = f"({', '.join(map(str, variable.dimensions))} "
            name_str = f'    {name:<{width}s} {dims_str} {variable.dtype}'
            lines.append(
                _pretty_print(f'{name_str}: {_dask_repr(variable.array)}'))
    # Attributes
    if len(ds.attrs):
        lines.append('  Attributes:')
        lines += _attributes_repr(ds.attrs)

    return '\n'.join(lines)


def _duplicate(variable: Variable, data: dask.array.core.Array) -> Variable:
    """Duplicate the variable with a new data.

    Args:
        variable: Variable to duplicate.
        data: The new data to use

    Returns:
        The duplicated variable
    """
    # pylint: disable=protected-access
    # _new is a protected member of this class
    return variable._new(variable.name, data, variable.dimensions,
                         variable.attrs, variable.compressor,
                         variable.fill_value, variable.filters)
    # pylint: enable=protected-access


class Dataset:
    """Hold variables, dimensions, and attributes that together form a dataset.

    Attrs:
        variables: Dataset variables
        attrs: Dataset attributes

    Raises:
        ValueError: If the dataset contains variables with the same dimensions
            but with different values.
    """
    __slots__ = ('dimensions', 'variables', 'attrs')

    def __init__(self,
                 variables: Iterable[Variable],
                 attrs: Sequence[Attribute] | None = None) -> None:
        #: The list of global attributes on this dataset
        self.attrs = attrs or []
        #: Dataset contents as dict of :py:class:`Variable` objects.
        self.variables = collections.OrderedDict(
            (item.name, item) for item in variables)
        #: A dictionary of dimension names and their index in the dataset
        self.dimensions: dict[str, int] = {}

        for var in self.variables.values():
            for ix, dim in enumerate(var.dimensions):
                if dim not in self.dimensions:
                    self.dimensions[dim] = var.array.shape[ix]
                elif self.dimensions[dim] != var.array.shape[ix]:
                    raise ValueError(f'variable {var.name} has conflicting '
                                     'dimensions')

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

    def __getstate__(self) -> tuple[Any, ...]:
        return self.dimensions, self.variables, self.attrs

    def __setstate__(self, state: tuple[Any, ...]) -> None:
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
                     data: ArrayLike[Any] | None = None):
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
                f'variable {variable.name} has dimensions '
                f'{variable.dimensions} that are not in the dataset')

        if data is None:
            shape = tuple(self.dimensions[dim] for dim in variable.dimensions)
            data = dask.array.wrap.full(shape,
                                        variable.fill_value,
                                        dtype=variable.dtype)
        else:
            for dim, size in zip(variable.dimensions, data.shape):
                if size != self.dimensions[dim]:
                    raise ValueError(
                        f'Conflicting sizes for dimension {dim!r}: '
                        f'length {self.dimensions[dim]} on the data but length '
                        f'{size} defined in dataset.')
        self.variables[variable.name] = Variable(
            variable.name,
            data,  # type: ignore[arg-type]
            variable.dimensions,
            variable.attrs,
            variable.compressor,
            variable.fill_value,
            variable.filters,
        )

    def rename(self, names: Mapping[str, str]) -> None:
        """Rename variables in the dataset.

        Args:
            names: A mapping from old names to new names

        Raises:
            ValueError: If the new names conflict with existing names
        """
        for old, new in names.items():
            if new in self.variables:
                raise ValueError(f'{new} already exists in the dataset')
            self.variables[new] = self.variables.pop(old).rename(new)

    def drops_vars(self, names: str | Sequence[str]) -> None:
        """Drop variables from the dataset.

        Args:
            names: Variable names to drop.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        # pylint: disable=expression-not-assigned
        {self.variables.pop(name) for name in names}
        # pylint: enable=expression-not-assigned

    def select_vars(self, names: str | Sequence[str]) -> Dataset:
        """Return a new dataset containing only the selected variables.

        Args:
            names: Variable names to select.

        Returns:
            A new dataset containing only the selected variables.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        return Dataset(
            [self.variables[name] for name in names],
            self.attrs,
        )

    def metadata(self) -> meta.Dataset:
        """Get the dataset metadata.

        Returns:
            Dataset metadata
        """
        return meta.Dataset(dimensions=tuple(self.dimensions.keys()),
                            variables=tuple(
                                item.metadata()
                                for item in self.variables.values()),
                            attrs=self.attrs)

    @staticmethod
    def from_xarray(ds: xarray.Dataset) -> Dataset:
        """Create a new dataset from an xarray dataset.

        Args:
            ds: Dataset to convert.

        Returns:
            New dataset.
        """
        variables = [
            Variable(
                name,  # type: ignore[arg-type]
                array.data,
                tuple(array.dims),  # type: ignore[arg-type]
                tuple(
                    Attribute(*attr)  # type: ignore[arg-type]
                    for attr in array.attrs.items()),
                array.encoding.get('compressor', None),
                array.encoding.get('_FillValue', None),
                array.encoding.get('filters', None))
            for name, array in ds.variables.items()
        ]

        return Dataset(
            variables=variables,
            attrs=tuple(
                Attribute(*item)  # type: ignore[arg-type]
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

    def to_dict(self,
                variables: Sequence[str] | None = None,
                **kwargs) -> dict[str, NDArray | NDMaskedArray]:
        """Convert the dataset to a dictionary, between the variable names and
        their data.

        Args:
            **kwargs: Additional parameters are passed through the function
                :py:func:`dask.compute`.

        Returns:
            Dictionary of variables.
        """
        variables = variables or tuple(self.variables.keys())
        arrays = tuple((key, value.data)
                       for key, value in self.variables.items()
                       if key in variables)
        return dict(dask.base.compute(*arrays, **kwargs))

    def isel(self, slices: dict[str, Any]) -> Dataset:
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
                f'Slices contain invalid dimension name(s): {dims_invalid}')
        default = slice(None)
        variables = [
            var.isel(tuple(slices.get(dim, default) for dim in var.dimensions))
            for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def delete(self, indexer: slice | Sequence[int], axis: str) -> Dataset:
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
            _duplicate(
                var,
                dask.array.routines.delete(var.array, indexer,
                                           var.dimensions.index(axis)))
            for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def compute(self, **kwargs) -> Dataset:
        """Compute the dataset variables.

        Args:
            **kwargs: Additional parameters are passed through to
                :py:func:`dask.array.compute`.

        Returns:
            New dataset.
        """
        arrays = tuple(item.array for item in self.variables.values())
        arrays = dask.base.compute(*arrays, **kwargs)

        variables = [
            _duplicate(self.variables[k], array)
            for k, array in zip(self.variables, arrays)
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def persist(self, **kwargs) -> Dataset:
        """Persist the dataset variables.

        Args:
            **kwargs: Additional parameters are passed to the function
                :py:func:`dask.array.Array.persist`.

        Returns:
            The dataset with the variables persisted into memory.
        """
        arrays = dask.base.persist(
            *tuple(item.data for item in self.variables.values()), **kwargs)
        for name, array in zip(self.variables, arrays):
            self.variables[name].array = array
        return self

    def concat(self, other: Dataset | Iterable[Dataset], dim: str) -> Dataset:
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
            raise ValueError('cannot concatenate an empty sequence')
        variables = [
            variable.concat(tuple(item.variables[name] for item in other), dim)
            for name, variable in self.variables.items()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)


def get_variable_metadata(variable: Variable | meta.Variable) -> meta.Variable:
    """Get the variable metadata.

    Args:
        variable: Variable to get the metadata for.

    Returns:
        Variable metadata.
    """
    if isinstance(variable, Variable):
        return variable.metadata()
    return variable
