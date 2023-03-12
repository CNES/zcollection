# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset
=======
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, OrderedDict, Sequence
import collections

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.array.rechunk
import dask.array.routines
import dask.array.wrap
import dask.base
import dask.threaded
import fsspec
import xarray

from . import meta, variable
from .compressed_array import CompressedArray
from .meta import Attribute, Dimension
from .type_hints import ArrayLike, NDArray, NDMaskedArray
from .variable import (
    Variable,
    _attributes_repr,
    _calculate_column_width,
    _dimensions_repr,
    _new_variable,
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
        for name, var in ds.variables.items():
            dims_str = f"({', '.join(map(str, var.dimensions))} "
            name_str = f'    {name:<{width}s} {dims_str} {var.dtype}'
            lines.append(_pretty_print(f'{name_str}: {_dask_repr(var.array)}'))
    # Attributes
    if len(ds.attrs):
        lines.append('  Attributes:')
        lines += _attributes_repr(ds.attrs)

    return '\n'.join(lines)


def _duplicate(var: variable.Variable,
               data: dask.array.core.Array) -> variable.Variable:
    """Duplicate the variable with a new data.

    Args:
        var: Variable to duplicate.
        data: The new data to use

    Returns:
        The duplicated variable
    """
    # pylint: disable=protected-access
    # _new is a protected member of this class
    return _new_variable(var.name, data, var.dimensions, var.attrs,
                         var.compressor, var.fill_value, var.filters)
    # pylint: enable=protected-access


class Dataset:
    """Hold variables, dimensions, and attributes that together form a dataset.

    Attrs:
        variables: Dataset variables
        attrs: Dataset attributes
        chunks: Chunk size for each dimension.
        block_size_limit: Maximum size (in bytes) of a
            block/chunk of variable's data.

    Raises:
        ValueError: If the dataset contains variables with the same dimensions
            but with different values.
    """
    __slots__ = ('dimensions', 'variables', 'attrs', 'chunks',
                 'block_size_limit')

    def __init__(self,
                 variables: Iterable[variable.Variable],
                 attrs: Sequence[Attribute] | None = None,
                 chunks: Sequence[Dimension] | None = None,
                 block_size_limit: int | None = None) -> None:
        #: The list of global attributes on this dataset
        self.attrs = tuple(attrs or [])
        #: Dataset contents as dict of
        #: :py:class:`Variable <zcollection.variable.Variable>` objects.
        self.variables = collections.OrderedDict(
            (item.name, item) for item in variables)
        #: A dictionary of dimension names and their index in the dataset
        self.dimensions: dict[str, int] = {}

        for var in self.variables.values():
            try:
                for ix, dim in enumerate(var.dimensions):
                    if dim not in self.dimensions:
                        self.dimensions[dim] = var.array.shape[ix]
                    elif self.dimensions[dim] != var.array.shape[ix]:
                        raise ValueError(
                            f'variable {var.name} has conflicting '
                            'dimensions')
            except IndexError as exc:
                raise ValueError(
                    f'variable {var.name} has missing dimensions') from exc

        #: Maximum data chunk size
        self.block_size_limit = block_size_limit or meta.BLOCK_SIZE_LIMIT

        #: Chunk size for each dimension
        chunks = chunks or []
        self.chunks: dict[str,
                          int | str] = {dim.name: dim.value
                                        for dim in chunks}

    def __len__(self) -> int:
        return len(self.variables)

    def __bool__(self) -> bool:
        return bool(self.variables)

    def __getitem__(self, name: str) -> variable.Variable:
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
        return (self.dimensions, self.variables, self.attrs, self.chunks,
                self.block_size_limit)

    def __setstate__(
        self, state: tuple[dict[str, int], OrderedDict[str, variable.Variable],
                           tuple[Attribute, ...], dict[str, int | str], int]
    ) -> None:
        (self.dimensions, self.variables, self.attrs, self.chunks,
         self.block_size_limit) = state

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes in the dataset.

        Returns:
            The total number of bytes in the dataset
        """
        return sum(item.nbytes for item in self.variables.values())

    @property
    def dims_chunk(self) -> Sequence[Dimension]:
        """Dimensions chunk size as a tuple.

        Returns:
            Dimensions associated to their chunk size.
        """
        return tuple(Dimension(*item) for item in self.chunks.items())

    def add_variable(self,
                     var: meta.Variable,
                     /,
                     data: ArrayLike[Any] | None = None):
        """Add a variable to the dataset.

        Args:
            var: The variable to add
            data: The data to add to the variable. If not provided, the variable
                will be created with the default fill value.

        Raises:
            ValueError: If the variable added has dimensions that conflict with
                existing dimensions, or if the variable has dimensions not
                defined in the dataset.
        """
        if set(var.dimensions) - set(self.dimensions):
            raise ValueError(f'variable {var.name} has dimensions '
                             f'{var.dimensions} that are not in the dataset')

        if data is None:
            shape = tuple(self.dimensions[dim] for dim in var.dimensions)
            data = dask.array.wrap.full(shape, var.fill_value, dtype=var.dtype)
        else:
            for dim, size in zip(var.dimensions, data.shape):
                if size != self.dimensions[dim]:
                    raise ValueError(
                        f'Conflicting sizes for dimension {dim!r}: '
                        f'length {self.dimensions[dim]} on the data but length '
                        f'{size} defined in dataset.')
        self.variables[var.name] = Variable(
            var.name,
            data,  # type: ignore[arg-type]
            var.dimensions,
            var.attrs,
            var.compressor,
            var.fill_value,
            var.filters,
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
        tuple(map(self.variables.pop, names))

    def select_vars(self, names: str | Sequence[str]) -> Dataset:
        """Return a new dataset containing only the selected variables.

        Args:
            names: Variable names to select.

        Returns:
            A new dataset containing only the selected variables.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        return Dataset([self.variables[name] for name in names],
                       self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    def metadata(self) -> meta.Dataset:
        """Get the dataset metadata.

        Returns:
            Dataset metadata
        """
        return meta.Dataset(
            dimensions=tuple(self.dimensions.keys()),
            variables=tuple(item.metadata()
                            for item in self.variables.values()),
            attrs=self.attrs,
            chunks=tuple(Dimension(*item) for item in self.chunks.items()),
            block_size_limit=self.block_size_limit)

    @staticmethod
    def from_xarray(ds: xarray.Dataset) -> Dataset:
        """Create a new dataset from a xarray dataset.

        Args:
            ds: Dataset to convert.

        Returns:
            New dataset.
        """
        variables = [
            Variable(
                name,  # type: ignore[arg-type]
                array.data,  # type: ignore[arg-type]
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
        """Convert the dataset to a xarray dataset.

        Args:
            **kwargs: Additional parameters are passed through the function
                :py:func:`xarray.conventions.decode_cf_variables`.

        Returns:
            Dataset as a xarray dataset.
        """
        data_vars = collections.OrderedDict(
            (name, var.to_xarray()) for name, var in self.variables.items())
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
            variables: Variables to include (default to all dataset's
                variables).
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

    def set_for_insertion(self, ds: meta.Dataset) -> Dataset:
        """Create a new dataset ready to be inserted into a collection.

        Args:
            ds: Dataset metadata.

        Returns:
            New dataset.
        """
        return Dataset(
            variables=[
                var.set_for_insertion() for var in self.variables.values()
            ],
            chunks=ds.chunks,
            block_size_limit=ds.block_size_limit,
        )

    def fill_attrs(self, ds: meta.Dataset):
        """Fill the dataset and its variables attributes using the provided
        metadata.

        Args:
            ds: Dataset metadata.
        """
        self.attrs = tuple(ds.attrs)
        _ = [
            var.fill_attrs(ds.variables[name])
            for name, var in self.variables.items()
        ]

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
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

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
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

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

        # Don't use _duplicate here because we want to transform
        # the numpy arrays computed by dask into dask arrays
        variables = [
            self.variables[k].duplicate(array)
            for k, array in zip(self.variables, arrays)
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    def rechunk(self, **kwargs) -> Dataset:
        """Rechunk the dataset.

        Args:
            **kwargs: Keyword arguments are passed through to
                :py:func:`dask.array.rechunk.rechunk`.

        Returns:
            New dataset.

        .. seealso:: :py:func:`dask.array.rechunk`
        """
        variables = [var.rechunk(**kwargs) for var in self.variables.values()]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    def persist(
        self,
        *,
        compress: bool = False,
        **kwargs,
    ) -> Dataset:
        """Persist the dataset variables.

        Args:
            compress: If true, compress the data loaded into memory.
            **kwargs: Additional parameters are passed to the function
                :py:func:`dask.array.Array.persist`.

        Returns:
            The dataset with the variables persisted into memory.
        """
        if compress:
            for var in self.variables.values():
                var.array = var.array.map_blocks(CompressedArray,
                                                 fill_value=var.fill_value)
        arrays = dask.base.persist(
            *tuple(item.data for item in self.variables.values()), **kwargs)
        variables = self.variables
        for name, array in zip(self.variables, arrays):
            variables[name].array = array

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
        if not isinstance(other, Iterable):
            other = [other]
        if not other:
            raise ValueError('cannot concatenate an empty sequence')
        variables = [
            var.concat(tuple(item.variables[name] for item in other), dim)
            for name, var in self.variables.items()
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    def merge(self, other: Dataset) -> None:
        """Merge the provided dataset into this dataset.

        Args:
            other: Dataset to merge into this dataset.
        """
        # Merge the variables
        for name, var in other.variables.items():

            # It's impossible to merge a variable with itself.
            if name in self.variables:
                raise ValueError(f'variable {name} already exists')
            self.variables[name] = var

        # If the dataset has common dimensions, they must be identical.
        same_dims = set(self.dimensions) & set(other.dimensions)
        if same_dims and not all(self.dimensions[dim] == other.dimensions[dim]
                                 for dim in same_dims):
            raise ValueError(f'dimensions {same_dims} are not identical')

        # Merge the dimensions.
        self.dimensions.update(other.dimensions)

        # Merge the attributes (overwriting any existing attributes).
        if self.attrs is None:
            self.attrs = other.attrs
        elif other.attrs is not None:
            attrs = dict(item.get_config() for item in self.attrs)
            attrs.update(item.get_config() for item in other.attrs)
            self.attrs = tuple(Attribute(*item) for item in attrs.items())

    def select_variables_by_dims(self,
                                 dims: Sequence[str],
                                 predicate: bool = True) -> Dataset:
        """Return a new dataset with only the variables that have the specified
        dimensions if predicate is true, otherwise return a new dataset with
        only the variables that do not have the specified dimensions.

        Args:
            dims: Dimensions to select.
            predicate: If true, select variables with the specified dimensions,
                otherwise select variables without the specified dimensions.

        Returns:
            New dataset or None if no variables match the predicate.
        """

        def _predicate_for_dimension_less(var: variable.Variable) -> bool:
            """Return true if the variable is selected by the predicate."""
            return (len(var.dimensions) == 0) == predicate

        def _predicate_for_dimension(var: variable.Variable) -> bool:
            """Return true if the variable is selected by the predicate."""
            return bool(set(var.dimensions) & set_of_dims) == predicate

        condition = (_predicate_for_dimension_less
                     if not dims else _predicate_for_dimension)

        set_of_dims = set(dims)
        variables = [var for var in self.variables.values() if condition(var)]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    def to_zarr(self,
                path: str,
                fs: fsspec.AbstractFileSystem | None = None,
                parallel: bool = True) -> None:
        """Write the dataset to a Zarr store.

        Args:
            path: Path to the Zarr store.
            fs: Filesystem to use.
            parallel: If true, write the data in parallel.
        """
        # pylint: disable=import-outside-toplevel, import-error
        # Avoid circular import
        import storage
        import sync

        # pylint: enable=import-outside-toplevel, import-error
        storage.write_zarr_group(self, path, fs or fsspec.filesystem('file'),
                                 sync.NoSync(), parallel)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)


def get_variable_metadata(
        var: variable.Variable | meta.Variable) -> meta.Variable:
    """Get the variable metadata.

    Args:
        var: Variable to get the metadata for.

    Returns:
        Variable metadata.
    """
    if isinstance(var, Variable):
        return var.metadata()
    return var
