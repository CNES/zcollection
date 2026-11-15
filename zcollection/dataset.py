# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset
=======
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import ast
import collections
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses

import dask.array.core
import dask.array.routines
import dask.array.wrap
import dask.base
import fsspec
import numpy
import xarray
import xarray.conventions

from . import meta, representation
from .compressed_array import CompressedArray
from .meta import BLOCK_SIZE_LIMIT, Attribute, Dimension, DimensionType
from .variable import Array, DelayedArray, Variable

if TYPE_CHECKING:
    import types

    from .type_hints import ArrayLike, NDArray, NDMaskedArray

#: Type alias for a dataset's variables.
VariableType = OrderedDict[str, Variable]

#: Type alias for a dataset's attributes.
AttributeType = tuple[Attribute, ...]


def _dask_repr(array: dask.array.core.Array) -> str:
    """Format a Dask array as a string.

    Args:
        array: The Dask array.

    Returns:
        The formatted string.
    """
    chunksize = tuple(item[0] for item in array.chunks)
    return f'dask.array<chunksize={chunksize}>'


def _dataset_repr(zds: Dataset) -> str:
    """Format a dataset as a string.

    Args:
        zds: The dataset.

    Returns:
        The formatted string.
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
    """Copy a variable, replacing its data.

    Args:
        var: The variable to copy.
        data: The data for the new variable.

    Returns:
        The copied variable.
    """
    return var.duplicate(data)


def _duplicate_array(var: Array, data: NDArray) -> Array:
    """Copy a variable, replacing its data.

    Args:
        var: The variable to copy.
        data: The data for the new variable.

    Returns:
        The copied variable.
    """
    return var.duplicate(data)


def _delete_delayed_vars(self: Dataset, indexer: slice | Sequence[int],
                         axis: str) -> list[Variable]:
    """Remove selected entries along the specified axis."""
    return [
        _duplicate_delayed_array(
            var,  # type: ignore[arg-type]
            dask.array.routines.delete(var.array, indexer,
                                       var.dimensions.index(axis)))
        for var in self.variables.values()
    ]


def _delete_vars(self: Dataset, indexer: slice | Sequence[int],
                 axis: str) -> list[Variable]:
    """Remove selected entries along the specified axis."""
    return [
        _duplicate_array(
            var,  # type: ignore[arg-type]
            numpy.delete(var.array, indexer, var.dimensions.index(axis)))
        for var in self.variables.values()
    ]


def _update_dimensions(self: Dataset,
                       delayed: bool | None = None) -> bool | None:
    """Update dataset dimension sizes from the variables."""
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
    """A dataset containing variables, dimensions, and attributes.

    A dataset stores either :py:class:`Array
    <zcollection.variable.array.Array>` or :py:class:`DelayedArray
    <zcollection.variable.delayed_array.DelayedArray>` variables. The two types
    cannot be mixed in the same dataset.

    Args:
        variables: An iterable of variable objects. The dataset indexes each
            variable by its ``name`` attribute.
        attrs: The global attributes of the dataset.
        delayed: Whether the dataset uses delayed variables. If ``None``, the
            value is inferred from its variables.

    Raises:
        ValueError: If variables assign different sizes to the same dimension,
            declare more dimensions than their data provides, or mix delayed
            and non-delayed arrays.
    """
    __slots__ = (
        'attrs',
        'block_size_limit',
        'delayed',
        'dim_chunks',
        'dimensions',
        'variables',
    )

    def __init__(self,
                 variables: Iterable[Variable],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 delayed: bool | None = None) -> None:
        #: Global dataset attributes.
        self.attrs = tuple(attrs or ())

        #: Dataset contents as dict of
        #: :py:class:`Variable <zcollection.variable.abc.Variable>` objects.
        #: If the dataset contains delayed variables, the values are
        #: :py:class:`DelayedArray
        #: <zcollection.variable.delayed_array.DelayedArray>` objects.
        #: Otherwise, the values are :py:class:`Array
        #: <zcollection.variable.array.Array>` objects.
        self.variables: collections.OrderedDict[
            str, Variable] = collections.OrderedDict(
                (item.name, item) for item in variables)

        #: Mapping of dimension names to their sizes.
        self.dimensions: DimensionType = {}

        # Update dimension sizes from the shape of each variable.
        delayed = _update_dimensions(self, delayed)

        #: Maximum data chunk size
        self.block_size_limit = BLOCK_SIZE_LIMIT

        #: Chunk size for each dimension
        self.dim_chunks: DimensionType = {}

        #: The representation used for dataset variables.
        self.delayed: bool = delayed if delayed is not None else True

    def __len__(self) -> int:
        return len(self.variables)

    def __bool__(self) -> bool:
        return bool(self.variables)

    def __getitem__(self, name: str) -> Variable:
        """Return a variable by name.

        Args:
            name: The name of the variable.

        Returns:
            The variable.

        Raises:
            KeyError: If the variable does not exist.
        """
        return self.variables[name]

    def __getattr__(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getstate__(self) -> tuple[Any, ...]:
        return (self.dimensions, self.variables, self.attrs, self.dim_chunks,
                self.block_size_limit, self.delayed)

    def __setstate__(
        self, state: tuple[DimensionType, VariableType, AttributeType,
                           DimensionType, int, bool]
    ) -> None:
        (self.dimensions, self.variables, self.attrs, self.dim_chunks,
         self.block_size_limit, self.delayed) = state

    @property
    def nbytes(self) -> int:
        """Return the total size of the dataset in bytes.

        Returns:
            The total number of bytes.
        """
        return sum(item.nbytes for item in self.variables.values())

    def add_variable(self,
                     var: meta.VariableSchema | Variable,
                     /,
                     data: ArrayLike[Any] | None = None,
                     filler: bool = False) -> None:
        """Add a variable to the dataset.

        Args:
            var: The variable to add.
            data: The data to use. Defaults to the data of ``var``, or to its
                fill value if no data is available.
            filler: Whether the variable was generated by the library.

        Raises:
            ValueError: If the variable introduces unknown dimensions or
                dimensions with incompatible sizes.
        """
        if set(var.dimensions) - set(self.dimensions):
            raise ValueError(f'variable {var.name} has dimensions '
                             f'{var.dimensions} that are not in the dataset')

        if data is None:
            # A variable provides both its metadata and its data.
            data = getattr(var, 'array', None)

        if data is None:
            shape = tuple(self.dimensions[dim] for dim in var.dimensions)
            data = dask.array.wrap.full(
                shape=shape, fill_value=var.fill_value,
                dtype=var.dtype) if self.delayed else numpy.full(
                    shape=shape, fill_value=var.fill_value, dtype=var.dtype)
        else:
            for dim, size in zip(var.dimensions, data.shape, strict=True):
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
            filler=filler)

    def rename(self, names: Mapping[str, str]) -> None:
        """Rename variables in the dataset.

        Args:
            names: A mapping from existing names to new names.

        Raises:
            ValueError: If a new name conflicts with an existing one.
        """
        for old, new in names.items():
            if new in self.variables:
                raise ValueError(f'{new} already exists in the dataset')
            self.variables[new] = self.variables.pop(old).rename(new)

    def drops_vars(self, names: str | Sequence[str]) -> None:
        """Remove variables from the dataset.

        Args:
            names: The names of the variables to remove.
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
        zds = Dataset((self.variables[name] for name in names),
                      attrs=self.attrs,
                      delayed=self.delayed)
        zds.copy_properties(ds=self)
        return zds

    def metadata(self) -> meta.DatasetSchema:
        """Return the dataset metadata.

        Returns:
            The dataset metadata.
        """
        return meta.DatasetSchema(
            tuple(
                Dimension(dim, size, self.dim_chunks.get(dim, -1))
                for dim, size in self.dimensions.items()),
            tuple(item.metadata() for item in self.variables.values()),
            attrs=self.attrs,
            block_size_limit=self.block_size_limit,
        )

    @staticmethod
    def from_xarray(zds: xarray.Dataset, delayed: bool = True) -> Dataset:
        """Create a dataset from an xarray dataset.

        Args:
            zds: The xarray dataset to convert.
            delayed: Whether to wrap the data in Dask arrays (``True``) or use
                NumPy arrays (``False``).

        Returns:
            The converted dataset.
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
            variables,
            attrs=tuple(
                Attribute(*item)  # type: ignore[arg-type]
                for item in zds.attrs.items()),
            delayed=delayed)

    def to_xarray(self, **kwargs) -> xarray.Dataset:
        """Convert the dataset to an xarray dataset.

        Args:
            **kwargs: Additional arguments passed to
                :py:func:`xarray.conventions.decode_cf_variables`.

        Returns:
            The xarray dataset.
        """
        data_vars = collections.OrderedDict(
            (name, var.to_xarray()) for name, var in self.variables.items())
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        decoded_vars, decoded_attrs, coord_names = (
            xarray.conventions.decode_cf_variables(data_vars, attrs, **kwargs))
        result = xarray.Dataset(decoded_vars, attrs=decoded_attrs)
        return result.set_coords(coord_names.intersection(decoded_vars))

    def to_dict(
        self,
        variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dict[str, NDArray | NDMaskedArray]:
        """Convert the selected variables to a dictionary of arrays.

        Args:
            variables: The variables to include. Defaults to all variables.
            **kwargs: Additional arguments passed to :py:func:`dask.compute`.

        Returns:
            A dictionary mapping variable names to their data.
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

    def copy_properties(self, ds: Dataset | meta.DatasetSchema):
        """Copy chunk and block settings from another dataset.

        Args:
            ds: The dataset providing these settings.
        """
        self.dim_chunks = ds.dim_chunks
        self.block_size_limit = ds.block_size_limit

    def fill_attrs(self, mds: meta.DatasetSchema) -> None:
        """Apply metadata attributes to the dataset and its variables.

        Args:
            mds: The dataset metadata.
        """
        self.attrs = tuple(mds.attrs)
        tuple(
            var.fill_attrs(mds.variables[  # type: ignore[func-returns-value]
                var.name]) for var in self.variables.values())

    def isel(self, slices: dict[str, Any]) -> Dataset:
        """Return a dataset selected by dimension indices.

        Args:
            slices: A mapping of dimension names to index slices.

        Returns:
            The selected dataset.
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
        zds = Dataset(variables, attrs=self.attrs, delayed=self.delayed)
        zds.copy_properties(ds=self)
        return zds

    def delete(self, indexer: slice | Sequence[int], axis: str) -> Dataset:
        """Return a dataset with the selected entries removed.

        Args:
            indexer: The indices to remove along ``axis``.
            axis: The dimension along which to remove entries.

        Returns:
            The resulting dataset.
        """
        variables: list[Variable] = _delete_delayed_vars(
            self, indexer, axis) if self.delayed else _delete_vars(
                self, indexer, axis)
        zds = Dataset(variables, attrs=self.attrs, delayed=self.delayed)

        zds.copy_properties(self)
        return zds

    def compute(self, **kwargs) -> Dataset:
        """Compute the dataset variables.

        Args:
            **kwargs: Additional arguments passed to :py:func:`dask.compute`.

        Returns:
            A dataset with computed variables, or the current dataset if its
            variables are already eager.
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
                  filters=item.filters,
                  filler=item.filler) for item, array in zip(
                      self.variables.values(), arrays, strict=True)
        ]
        zds = Dataset(variables, attrs=self.attrs, delayed=False)

        zds.copy_properties(ds=self)
        return zds

    def rechunk(self, **kwargs) -> Dataset:
        """Rechunk the dataset variables.

        Args:
            **kwargs: Arguments passed to :py:func:`dask.array.rechunk`.

        Returns:
            The rechunked dataset, or the current dataset if its variables
            are already eager.

        .. seealso:: :py:func:`dask.array.rechunk`
        """
        if not self.delayed:
            return self
        variables: list[Variable] = [
            var.rechunk(**kwargs) for var in self.variables.values()
        ]
        zds = Dataset(variables, attrs=self.attrs, delayed=self.delayed)
        zds.copy_properties(ds=self)
        return zds

    def persist(
        self,
        *,
        compress: bool = False,
        **kwargs,
    ) -> Dataset:
        """Persist the dataset variables in memory.

        If ``compress`` is enabled, persisted blocks are stored as compressed
        arrays. The method updates and returns the current dataset.

        Args:
            compress: Whether to compress the persisted data in memory.
            **kwargs: Additional arguments passed to
                :py:meth:`dask.array.Array.persist`.

        Returns:
            The dataset with persisted variables.
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
        for name, array in zip(self.variables, arrays, strict=True):
            variables[name].array = array

        return self

    def concat(self, other: Dataset | Iterable[Dataset], dim: str) -> Dataset:
        """Concatenate datasets along a dimension.

        Args:
            other: The datasets to concatenate.
            dim: The dimension along which to concatenate.

        Returns:
            The concatenated dataset.

        Raises:
            ValueError: If the sequence is empty, the datasets use different
                array representations, or they have different variables.
        """
        if not isinstance(other, Iterable):
            other = [other]
        other = tuple(other)
        if not other:
            raise ValueError('cannot concatenate an empty sequence')
        if not all(item.delayed == self.delayed for item in other):
            raise ValueError('cannot concatenate delayed and non-delayed data')
        names: set[str] = set(self.variables)
        differing: set[str] = set().union(*(names ^ set(item.variables)
                                            for item in other))
        if differing:
            raise ValueError('cannot concatenate datasets holding different '
                             f'variables: {", ".join(sorted(differing))}')
        variables: list[Variable] = [
            var.concat(tuple(item.variables[name] for item in other), dim)
            for name, var in self.variables.items()
        ]
        zds = Dataset(variables, attrs=self.attrs, delayed=self.delayed)
        zds.copy_properties(ds=self)
        return zds

    def merge(self, other: Dataset | None) -> None:
        """Merge another dataset's variables, dimensions, and attributes into
        this dataset.

        Args:
            other: The dataset to merge.
        """
        if other is None:
            return

        if self.delayed != other.delayed:
            raise ValueError('cannot merge delayed and non-delayed data')

        # Merge the variables
        for name, var in other.variables.items():

            # Reject variable names that already exist in the dataset.
            if name in self.variables:
                raise ValueError(f'variable {name} already exists')
            self.variables[name] = var

        # Shared dimensions must have the same size.
        same_dims: set[str] = set(self.dimensions) & set(other.dimensions)
        if same_dims and not all(self.dimensions[dim] == other.dimensions[dim]
                                 for dim in same_dims):
            raise ValueError(f'dimensions {same_dims} are not identical')

        # Merge the dimensions.
        self.dimensions.update(other.dimensions)

        # Merge attributes, replacing existing values when names overlap.
        if self.attrs is None:
            self.attrs = other.attrs
        elif other.attrs is not None:
            attrs = dict(item.get_config() for item in self.attrs)
            attrs.update(item.get_config() for item in other.attrs)
            self.attrs = tuple(Attribute(*item) for item in attrs.items())

    def select_variables_by_dims(self,
                                 dims: Sequence[str],
                                 predicate: bool = True) -> Dataset:
        """Select variables according to their dimensions.

        When ``predicate`` is ``True``, keep variables sharing at least one
        dimension with ``dims``. If ``dims`` is empty, keep variables without
        dimensions instead. When ``predicate`` is ``False``, keep the other
        variables. An empty selection returns an empty dataset.

        Args:
            dims: The dimensions used for selection.
            predicate: Whether to select matching (``True``) or nonmatching
                (``False``) variables.

        Returns:
            A new dataset containing the selected variables.
        """

        def _predicate_for_dimension_less(var: Variable) -> bool:
            """Return whether the variable matches the selection predicate."""
            return (len(var.dimensions) == 0) == predicate

        def _predicate_for_dimension(var: Variable) -> bool:
            """Return whether the variable matches the selection predicate."""
            return bool(set(var.dimensions) & set_of_dims) == predicate

        condition: Callable[..., bool]
        condition = (_predicate_for_dimension_less
                     if not dims else _predicate_for_dimension)

        set_of_dims = set(dims)
        variables: list[Variable] = [
            var for var in self.variables.values() if condition(var)
        ]
        zds = Dataset(variables, attrs=self.attrs)
        zds.copy_properties(self)
        return zds

    @classmethod
    def from_zarr(cls,
                  path: str,
                  fs: fsspec.AbstractFileSystem | None = None,
                  *,
                  delayed: bool = True) -> Dataset:
        """Read a dataset from a Zarr group.

        Args:
            path: The group to read.
            fs: The file system containing the group. Defaults to the local
                file system.
            delayed: Whether to load the variables lazily. Defaults to
                ``True``.

        Returns:
            The dataset holding the variables and attributes of the group.
        """
        # Avoid circular import
        from . import storage

        return storage.open_zarr_group(path,
                                       fs or fsspec.filesystem('file'),
                                       delayed=delayed)

    def to_zarr(self,
                path: str,
                fs: fsspec.AbstractFileSystem | None = None,
                parallel: bool = True) -> None:
        """Write the dataset to a Zarr group.

        Args:
            path: The destination group.
            fs: The file system for the destination. Defaults to the local
                file system.
            parallel: Whether to write the variables in parallel with the
                current Dask scheduler. Defaults to ``True``.
        """
        # Avoid circular import
        from . import storage

        storage.write_dataset(self,
                              path,
                              fs or fsspec.filesystem('file'),
                              parallel=parallel)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)


def get_variable_metadata(
        var: Variable | meta.VariableSchema) -> meta.VariableSchema:
    """Return the metadata of a variable.

    Args:
        var: The variable.

    Returns:
        Its metadata.

    Raises:
        TypeError: If the input is not a supported variable type.
    """
    if isinstance(var, Variable):
        return var.metadata()
    if not isinstance(var, meta.VariableSchema):
        raise TypeError('Expected a meta.VariableSchema or a Variable, got '
                        f'{type(var).__name__}')
    return var


def get_dataset_variable_properties(
        metadata: meta.DatasetSchema,
        selected_variables: Iterable[str] | None = None) -> tuple[Array, ...]:
    """Create empty variables using a dataset schema.

    Args:
        metadata: The dataset schema.
        selected_variables: The variables to include. If ``None`` or empty,
            all variables are included.

    Returns:
        The variables with empty arrays and the properties from the schema.
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


@dataclasses.dataclass
class Expression:
    """Evaluate an expression using variables from a dataset.

    The expression can use the supported NumPy functions and constants.

    Args:
        expression: The expression to evaluate.

    Raises:
        NameError: If a referenced variable is missing from the dataset
            when the expression is evaluated.

    Example:
        >>> expr = Expression("a + b * sin(c)")
    """
    #: Compiled expression to be evaluated
    code: types.CodeType

    #: Names of the instance attributes defined by ``__slots__``.
    __slots__: tuple[str, ...] = ('code', )

    #: The builtins that are allowed in the expression.
    BUILTINS: ClassVar[dict[str, Any]] = {
        'sin': numpy.sin,
        'cos': numpy.cos,
        'tan': numpy.tan,
        'pi': numpy.pi,
        'e': numpy.e,
        'exp': numpy.exp,
        'log': numpy.log,
        'log10': numpy.log10,
        'log2': numpy.log2,
        'sqrt': numpy.sqrt,
        'arcsin': numpy.arcsin,
        'arccos': numpy.arccos,
        'arctan': numpy.arctan,
        'arctan2': numpy.arctan2,
        'sinh': numpy.sinh,
        'cosh': numpy.cosh,
        'tanh': numpy.tanh,
        'arcsinh': numpy.arcsinh,
        'arccosh': numpy.arccosh,
        'arctanh': numpy.arctanh,
        'ceil': numpy.ceil,
        'floor': numpy.floor,
        'round': numpy.round,
        'abs': numpy.abs,
        'min': numpy.min,
        'max': numpy.max,
        'sum': numpy.sum,
        'prod': numpy.prod,
        'mean': numpy.mean,
        'std': numpy.std,
        'var': numpy.var,
        'median': numpy.median,
        'percentile': numpy.percentile,
        'power': numpy.power,
        'mod': numpy.mod,
        'fmod': numpy.fmod,
        'remainder': numpy.remainder,
        'isfinite': numpy.isfinite,
        'isinf': numpy.isinf,
        'isnan': numpy.isnan,
        'isnat': numpy.isnat,
    }

    def __init__(self, expression: str) -> None:
        self.code = compile(ast.parse(expression, mode='eval'), ' ', 'eval')

    def __call__(self, zds: Dataset) -> dask.array.core.Array:
        try:
            __locals: dict[str, Any] = {
                name: zds.variables[name].data
                for name in self.code.co_names if name not in self.BUILTINS
            }

            # Evaluate using the supported NumPy functions and constants.
            return eval(self.code, {'__builtins__': self.BUILTINS}, __locals)

        except KeyError as exc:
            raise NameError(
                f'Variable {exc} is not defined in the dataset') from exc
