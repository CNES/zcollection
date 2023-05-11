# Copyright (c) 2023 CNES
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
    Callable,
    Dict,
    Iterable,
    Mapping,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)
import collections

import dask.array.core
import dask.array.routines
import dask.array.wrap
import dask.base
import numpy
import xarray
import xarray.conventions

from . import meta, representation
from .compressed_array import CompressedArray
from .meta import BLOCK_SIZE_LIMIT, Attribute, Dimension
from .type_hints import ArrayLike, NDArray, NDMaskedArray
from .variable import Array, DelayedArray, Variable, new_variable

#: Alias to type hint for the dimensions of a dataset.
DimensionType = Dict[str, int]

#: Alias to type hint for the variables of a dataset.
VariableType = OrderedDict[str, Variable]

#: Alias to type hint for the attributes of a dataset.
AttributeType = Tuple[Attribute, ...]

#: Alias to type hint for the chunk sizes for each dimension of a dataset.
ChunkType = Dict[str, Union[int, str]]


def _dask_repr(array: dask.array.core.Array) -> str:
    """Get the string representation of a dask array.

    Args:
        array: A dask array.

    Returns:
        The string representation of the dask array.
    """
    chunksize = tuple(item[0] for item in array.chunks)
    return f'dask.array<chunksize={chunksize}>'


def _dataset_repr(zds: Dataset) -> str:
    """Get the string representation of a dataset.

    Args:
        zds: A dataset.

    Returns:
        The string representation of the dataset.
    """
    # Dimensions
    dims_str: str = representation.dimensions(zds.dimensions)
    lines: list[str] = [
        f'<{zds.__module__}.{zds.__class__.__name__}>',
        f'  Dimensions: {dims_str}', 'Data variables:'
    ]
    # Variables
    if len(zds.variables) == 0:
        lines.append('    <empty>')
    else:
        width: int = representation.calculate_column_width(zds.variables)
        for name, var in zds.variables.items():
            dims_str = f"({', '.join(map(str, var.dimensions))}) "
            name_str: str = f'    {name:<{width}s} {dims_str} {var.dtype}'
            data_str: str = _dask_repr(var.data) if zds.delayed else '...'
            lines.append(
                representation.pretty_print(f'{name_str}: {data_str}'))
    # Attributes
    if len(zds.attrs):
        lines.append('  Attributes:')
        lines += representation.attributes(zds.attrs)

    return '\n'.join(lines)


def _duplicate_delayed_array(var: DelayedArray,
                             data: dask.array.core.Array) -> DelayedArray:
    """Duplicate the variable with a new data.

    Args:
        var: Variable to duplicate.
        data: The new data to use

    Returns:
        The duplicated variable
    """
    return new_variable(type(var),
                        name=var.name,
                        array=data,
                        dimensions=var.dimensions,
                        attrs=var.attrs,
                        compressor=var.compressor,
                        fill_value=var.fill_value,
                        filters=var.filters)


def _duplicate_array(var: Array, data: NDArray) -> Array:
    """Duplicate the variable with a new data.

    Args:
        var: Variable to duplicate.
        data: The new data to use

    Returns:
        The duplicated variable
    """
    return new_variable(type(var),
                        name=var.name,
                        array=data,
                        dimensions=var.dimensions,
                        attrs=var.attrs,
                        compressor=var.compressor,
                        fill_value=var.fill_value,
                        filters=var.filters)


def _delete_delayed_vars(self: Dataset, indexer: slice | Sequence[int],
                         axis: str) -> list[Variable]:
    """Delete the variables along the given axis."""
    return [
        _duplicate_delayed_array(
            var,  # type: ignore[arg-type]
            dask.array.routines.delete(var.array, indexer,
                                       var.dimensions.index(axis)))
        for var in self.variables.values()
    ]


def _delete_vars(self: Dataset, indexer: slice | Sequence[int],
                 axis: str) -> list[Variable]:
    """Delete the variables along the given axis."""
    return [
        _duplicate_array(
            var,  # type: ignore[arg-type]
            numpy.delete(var.array, indexer, var.dimensions.index(axis)))
        for var in self.variables.values()
    ]


def _update_dimensions(self: Dataset,
                       delayed: bool | None = None) -> bool | None:
    """Update the dimensions of the dataset based on its variables."""
    for var in self.variables.values():
        if delayed is None:
            delayed = isinstance(var, DelayedArray)
        elif delayed != isinstance(var, DelayedArray):
            raise ValueError(
                'the dataset contains both delayed and non-delayed '
                'variables')
        try:
            for idx, dim in enumerate(var.dimensions):
                if dim not in self.dimensions:
                    self.dimensions[dim] = var.array.shape[idx]
                elif self.dimensions[dim] != var.array.shape[idx]:
                    raise ValueError(f'variable {var.name} has conflicting '
                                     'dimensions')
        except IndexError as exc:
            raise ValueError(
                f'variable {var.name} has missing dimensions') from exc
    return delayed


class Dataset:
    """Hold variables, dimensions, and attributes that together form a dataset.

    Args:
        variables (DelayedArray | Array): A dictionary of variables in the
            dataset, with variable names as keys and :py:class:`Array
            <zcollection.variable.array.Array>` or :py:class:`DelayedArray
            <zcollection.variable.delayed.DelayedArray>` objects as values.
        attrs: A tuple of global attributes on this dataset.
        block_size_limit: The maximum size (in bytes) of a block/chunk of
            variable's data. Defaults to 128 MiB.
        chunks: A dictionary of chunk sizes for each dimension.
        delayed: A boolean indicating whether the dataset contains delayed
            variables (numpy arrays wrapped in dask arrays).

    Raises:
        ValueError: If the dataset contains variables with the same dimensions
            but with different values.
        ValueError: If the dataset contains both delayed and non-delayed
            variables.

    Notes:
        The dataset is a dictionary-like container of variables. It also holds
        the dimensions and attributes of the dataset.
        If the dataset contains delayed variables, the values are
        :py:class:`DelayedArray
        <zcollection.variable.delayed_array.DelayedArray>` objects. Otherwise,
        the values are :py:class:`Array
        <zcollection.variable.array.Array>` objects. It is impossible to mix
        delayed and non-delayed variables in the same dataset.
    """
    __slots__ = ('dimensions', 'variables', 'attrs', 'chunks',
                 'block_size_limit', 'delayed')

    def __init__(self,
                 variables: Iterable[Variable],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 block_size_limit: int | None = None,
                 chunks: Sequence[Dimension] | None = None,
                 delayed: bool | None = None) -> None:
        #: The list of global attributes on this dataset
        self.attrs = tuple(attrs or ())

        #: Dataset contents as dict of :py:class:`Variable
        #: <zcollection.variable.abc.Variable>` objects. If the dataset
        #: contains delayed variables, the values are :py:class:`DelayedArray
        #: <zcollection.variable.delayed_array.DelayedArray>` objects.
        #: Otherwise, the values are :py:class:`Array
        #: <zcollection.variable.array.Array>` objects.
        self.variables = collections.OrderedDict(
            (item.name, item) for item in variables)

        #: A dictionary of dimension names and their index in the dataset
        self.dimensions: DimensionType = {}

        # Loops over each variable in the dataset and updates the dimensions
        # according to the shape of the array.
        delayed = _update_dimensions(self, delayed)

        #: Maximum data chunk size
        self.block_size_limit: int = block_size_limit or BLOCK_SIZE_LIMIT

        #: Chunk size for each dimension
        self.chunks: ChunkType = {dim.name: dim.value for dim in chunks or []}

        #: The type of variables in the dataset
        self.delayed: bool = delayed if delayed is not None else True

    def __len__(self) -> int:
        return len(self.variables)

    def __bool__(self) -> bool:
        return bool(self.variables)

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

    def __getattr__(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self) -> tuple[Any, ...]:
        return (self.dimensions, self.variables, self.attrs, self.chunks,
                self.block_size_limit, self.delayed)

    def __setstate__(
        self, state: tuple[DimensionType, VariableType, AttributeType,
                           ChunkType, int, bool]
    ) -> None:
        (self.dimensions, self.variables, self.attrs, self.chunks,
         self.block_size_limit, self.delayed) = state

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
                     data: ArrayLike[Any] | None = None) -> None:
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
            data = dask.array.wrap.full(
                shape, var.fill_value,
                dtype=var.dtype) if self.delayed else numpy.full(
                    shape, var.fill_value, dtype=var.dtype)
        else:
            for dim, size in zip(var.dimensions, data.shape):
                if size != self.dimensions[dim]:
                    raise ValueError(
                        f'Conflicting sizes for dimension {dim!r}: '
                        f'length {self.dimensions[dim]} on the data but length '
                        f'{size} defined in dataset.')
        self.variables[var.name] = (DelayedArray if self.delayed else Array)(
            var.name,
            data,
            var.dimensions,
            attrs=var.attrs,
            compressor=var.compressor,
            fill_value=var.fill_value,
            filters=var.filters,
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
        return Dataset((self.variables[name] for name in names),
                       attrs=self.attrs,
                       block_size_limit=self.block_size_limit,
                       chunks=self.dims_chunk)

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
            block_size_limit=self.block_size_limit,
        )

    @staticmethod
    def from_xarray(zds: xarray.Dataset, delayed: bool = True) -> Dataset:
        """Create a new dataset from a xarray dataset.

        Args:
            zds: Dataset to convert.
            delayed: If True, the data will be wrapped in a dask array. If
                False, the data will be handled as a numpy array.

        Returns:
            New dataset.
        """
        handler: type[DelayedArray | Array]
        handler = DelayedArray if delayed else Array
        variables: list[Variable] = [
            handler(
                name,  # type: ignore[arg-type]
                array.data,  # type: ignore[arg-type]
                tuple(array.dims),  # type: ignore[arg-type]
                attrs=tuple(
                    Attribute(*attr)  # type: ignore[arg-type]
                    for attr in array.attrs.items()),
                compressor=array.encoding.get('compressor', None),
                fill_value=array.encoding.get('_FillValue', None),
                filters=array.encoding.get('filters', None))
            for name, array in zds.variables.items()
        ]

        return Dataset(
            variables=variables,
            attrs=tuple(
                Attribute(*item)  # type: ignore[arg-type]
                for item in zds.attrs.items()),
            delayed=delayed,
        )

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
        result = xarray.Dataset(data_vars, attrs=attrs)
        return result.set_coords(coord_names.intersection(data_vars))

    def to_dict(
        self,
        variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dict[str, NDArray | NDMaskedArray]:
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
        if self.delayed:
            arrays = tuple((key, value.data)
                           for key, value in self.variables.items()
                           if key in variables)
            return dict(dask.base.compute(*arrays, **kwargs))
        return dict(
            tuple((key, value.values) for key, value in self.variables.items()
                  if key in variables))

    def set_for_insertion(self, mds: meta.Dataset) -> Dataset:
        """Create a new dataset ready to be inserted into a collection.

        Args:
            mds: Dataset metadata.

        Returns:
            New dataset.
        """
        return Dataset(
            variables=[
                var.set_for_insertion() for var in self.variables.values()
            ],
            chunks=mds.chunks,
            block_size_limit=mds.block_size_limit,
        )

    def fill_attrs(self, mds: meta.Dataset) -> None:
        """Fill the dataset and its variables attributes using the provided
        metadata.

        Args:
            mds: Dataset metadata.
        """
        self.attrs = tuple(mds.attrs)
        tuple(
            map(lambda var: var.fill_attrs(mds.variables[var.name]),
                self.variables.values()))

    def isel(self, slices: dict[str, Any]) -> Dataset:
        """Return a new dataset with each array indexed along the specified
        slices.

        Args:
            slices: Dictionary of dimension names and slices

        Returns:
            New dataset.
        """
        dims_invalid: set[str] = set(slices) - set(self.dimensions)
        if dims_invalid:
            raise ValueError(
                f'Slices contain invalid dimension name(s): {dims_invalid}')
        variables: list[Variable] = [
            var.isel(
                tuple(slices.get(dim, slice(None)) for dim in var.dimensions))
            for var in self.variables.values()
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       block_size_limit=self.block_size_limit,
                       chunks=self.dims_chunk,
                       delayed=self.delayed)

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
        variables: list[Variable] = _delete_delayed_vars(
            self, indexer, axis) if self.delayed else _delete_vars(
                self, indexer, axis)
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       block_size_limit=self.block_size_limit,
                       chunks=self.dims_chunk,
                       delayed=self.delayed)

    def compute(self, **kwargs) -> Dataset:
        """Compute the dataset variables.

        Args:
            **kwargs: Additional parameters are passed through to
                :py:func:`dask.array.compute`.

        Returns:
            New dataset.
        """
        if not self.delayed:
            return self

        arrays: Iterable[NDArray] = dask.base.compute(
            *tuple(item.array for item in self.variables.values()), **kwargs)

        variables: list[Variable] = [
            Array(item.name,
                  array,
                  item.dimensions,
                  attrs=item.attrs,
                  compressor=item.compressor,
                  fill_value=item.fill_value,
                  filters=item.filters)
            for item, array in zip(self.variables.values(), arrays)
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit,
                       delayed=False)

    def rechunk(self, **kwargs) -> Dataset:
        """Rechunk the dataset.

        Args:
            **kwargs: Keyword arguments are passed through to
                :py:func:`dask.array.rechunk.rechunk`.

        Returns:
            New dataset.

        .. seealso:: :py:func:`dask.array.rechunk`
        """
        if not self.delayed:
            return self
        variables: list[Variable] = [
            var.rechunk(**kwargs) for var in self.variables.values()
        ]
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
        if not self.delayed:
            return self
        if compress:
            for var in self.variables.values():
                var.array = var.array.map_blocks(CompressedArray,
                                                 fill_value=var.fill_value)
        arrays: Iterable[NDArray] = dask.base.persist(
            *tuple(item.data for item in self.variables.values()), **kwargs)
        variables: OrderedDict[str, Variable] = self.variables
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
        other = tuple(other)
        if not other:
            raise ValueError('cannot concatenate an empty sequence')
        if not all(item.delayed == self.delayed for item in other):
            raise ValueError('cannot concatenate delayed and non-delayed data')
        variables: list[Variable] = [
            var.concat(tuple(item.variables[name] for item in other), dim)
            for name, var in self.variables.items()
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       block_size_limit=self.block_size_limit,
                       chunks=self.dims_chunk,
                       delayed=self.delayed)

    def merge(self, other: Dataset) -> None:
        """Merge the provided dataset into this dataset.

        Args:
            other: Dataset to merge into this dataset.
        """
        if self.delayed != other.delayed:
            raise ValueError('cannot merge delayed and non-delayed data')

        # Merge the variables
        for name, var in other.variables.items():

            # It's impossible to merge a variable with itself.
            if name in self.variables:
                raise ValueError(f'variable {name} already exists')
            self.variables[name] = var

        # If the dataset has common dimensions, they must be identical.
        same_dims: set[str] = set(self.dimensions) & set(other.dimensions)
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

        def _predicate_for_dimension_less(var: Variable) -> bool:
            """Return true if the variable is selected by the predicate."""
            return (len(var.dimensions) == 0) == predicate

        def _predicate_for_dimension(var: Variable) -> bool:
            """Return true if the variable is selected by the predicate."""
            return bool(set(var.dimensions) & set_of_dims) == predicate

        condition: Callable[..., bool]
        condition = (_predicate_for_dimension_less
                     if not dims else _predicate_for_dimension)

        set_of_dims = set(dims)
        variables: list[Variable] = [
            var for var in self.variables.values() if condition(var)
        ]
        return Dataset(variables=variables,
                       attrs=self.attrs,
                       chunks=self.dims_chunk,
                       block_size_limit=self.block_size_limit)

    # def to_zarr(self,
    #             path: str,
    #             fs: fsspec.AbstractFileSystem | None = None,
    #             parallel: bool = True) -> None:
    #     """Write the dataset to a Zarr store.

    #     Args:
    #         path: Path to the Zarr store.
    #         fs: Filesystem to use.
    #         parallel: If true, write the data in parallel.
    #     """
    #     if not self.delayed:
    #         raise ValueError('cannot write a non-delayed dataset to Zarr')
    #     # pylint: disable=import-outside-toplevel, import-error
    #     # Avoid circular import
    #     import storage
    #     import sync

    #     # pylint: enable=import-outside-toplevel, import-error
    #     storage.write_zarr_group(self, path, fs or fsspec.filesystem('file'),
    #                              sync.NoSync(), parallel)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)


def get_variable_metadata(var: Variable | meta.Variable) -> meta.Variable:
    """Get the variable metadata.

    Args:
        var: Variable to get the metadata for.

    Returns:
        Variable metadata.
    """
    if isinstance(var, Variable):
        return var.metadata()
    return var


def get_dataset_variable_properties(
        metadata: meta.Dataset,
        selected_variables: Iterable[str] | None = None) -> tuple[Array, ...]:
    """Return the variables properties defined in the dataset.

    Args:
        metadata: Metadata dataset containing variables information.
        selected_variables: The variables to return. If None, all the
            variables are returned.

    Returns:
        The variables defined in the dataset.
    """
    selected_variables = selected_variables or metadata.variables.keys()
    return tuple(
        Array(v.name,
              numpy.ndarray((0, ) * len(v.dimensions), v.dtype),
              v.dimensions,
              attrs=v.attrs,
              compressor=v.compressor,
              fill_value=v.fill_value,
              filters=v.filters) for k, v in metadata.variables.items()
        if k in selected_variables)
