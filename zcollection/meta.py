# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Configuration metadata
======================
"""
from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Sequence

import numcodecs.abc
import numpy
import zarr.codecs
import zarr.meta

from .type_hints import DTypeLike

#: Block size limit used with dask arrays. (128 MiB)
BLOCK_SIZE_LIMIT = 134217728

#: Alias to type hint for the dimensions of a dataset.
DimensionType = dict[str, int]


class Attribute:
    """Handle the metadata of a dataset attribute.

    Args:
        name: name of the attribute.
        value: value of the attribute.
    """
    __slots__ = ('name', 'value')

    def __init__(self, name: str, value: Any) -> None:
        #: Name of the key.
        self.name: str = name
        #: Value of the key.
        self.value: Any = self._encode(value)

    @staticmethod
    def _encode(value: Any) -> Any:
        """Encode an attribute value as something that can be serialized as
        JSON."""
        if isinstance(value, numpy.ndarray):
            return value.tolist()

        if isinstance(value, numpy.generic):
            return value.item()

        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!r}, {self.value})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Attribute):
            return False
        return self.get_config() == other.get_config()

    def get_config(self) -> tuple[str, Any]:
        """Get the key/value pair configuration."""
        return self.name, self.value

    @staticmethod
    def from_config(data: tuple[str, Any]) -> Attribute:
        """Create a new instance from its metadata.

        Args:
            data: attribute configuration.

        Returns:
            Attribute: a new attribute.
        """
        return Attribute(*data)


class Dimension:
    """Handle the metadata of a dataset dimension.

    Args:
        name: name of the dimension.
        value: size of the dimension.
        chunks: size of the dimension's chunks.
    """
    __slots__ = ('name', 'value', 'chunks')

    def __init__(self, name: str, value: int, chunks: int = -1) -> None:
        self.name: str = name
        self.value: int = value
        self.chunks: int = chunks

    @staticmethod
    def from_config(data: tuple[str, int, int]) -> Dimension:
        """Creates a new instance from its metadata.

        Returns:
            Dimension: a new dimension.
        """
        return Dimension(*data)

    def get_config(self) -> tuple[str, int, int]:
        """Get the key/value/chunk values configuration."""
        return self.name, self.value, self.chunks

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.name!r}, '
                f'{self.value}, {self.chunks})')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return self.get_config() == other.get_config()


class Variable:
    """Handle the metadata of a dataset variable.

    Args:
        name: Name of the variable.
        dtype: Data type of the variable.
        dimensions: Names of the dimensions of the variable. Defaults to None.
        attrs: Attributes of the variable. Defaults to None.
        compressor: Compression codec for the variable. Defaults to None.
        fill_value: Fill value for the variable. Defaults to None.
        filters: Filters for the variable. Defaults to None.

    Warning:
        If the variable uses filters, the ``fill_value`` parameter must be the
        value that results from decoding the filter. For example, if the filter
        is ``FixedScaleOffset(0, 1000)`` and the desired ``fill_value`` is
        ``65536``, then the ``fill_value`` parameter must be ``65536 / 1000 =
        65.536``.
    """
    __slots__ = ('attrs', 'compressor', 'dimensions', 'dtype', 'fill_value',
                 'filters', 'name')

    def __init__(self,
                 name: str,
                 dtype: DTypeLike,
                 *,
                 dimensions: Sequence[str] | None = None,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        attrs = attrs or ()

        #: Attributes of the variable.
        self.attrs = tuple(attrs)
        #: Compression codec for the variable.
        self.compressor: numcodecs.abc.Codec | None = compressor
        #: Dimensions of the variable.
        self.dimensions = tuple(dimensions or ())
        #: Data type of the variable.
        self.dtype = numpy.dtype(dtype)
        #: Fill value for the variable.
        self.fill_value: Any | None = fill_value
        #: Filter codecs for the variable.
        self.filters = tuple(filters or ())
        #: Variable name.
        self.name: str = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.get_config() == other.get_config()

    def get_config(self) -> dict[str, Any]:
        """Get the variable metadata.

        Returns:
            variable configuration.
        """
        compressor: numcodecs.abc.Codec | None
        compressor_config: dict[str, None] | None

        compressor = self.compressor
        compressor_config = compressor.get_config(
        ) if compressor is not None else None

        return {
            'attrs': sorted(item.get_config() for item in self.attrs),
            'compressor': compressor_config,
            'dimensions': self.dimensions,
            'dtype': zarr.meta.encode_dtype(self.dtype),
            'fill_value': zarr.meta.encode_fill_value(self.fill_value,
                                                      self.dtype),
            'filters': tuple(item.get_config() for item in self.filters),
            'name': self.name,
        }

    @staticmethod
    def from_config(data: dict[str, Any]) -> Variable:
        """Create a new variable from the given variable configuration.

        Args:
            data: variable configuration.

        Returns:
            new variable.
        """

        def get_codec(codec) -> numcodecs.abc.Codec | None:
            """Get the codec from its configuration."""
            return zarr.codecs.get_codec(codec) if codec is not None else None

        dtype: DTypeLike = zarr.meta.decode_dtype(data['dtype'])
        filters: Sequence[numcodecs.abc.Codec] = tuple(
            zarr.codecs.get_codec(item) for item in data['filters']
            if item is not None)

        return Variable(
            data['name'],
            dtype,
            dimensions=data['dimensions'],
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            compressor=get_codec(data['compressor']),
            fill_value=zarr.meta.decode_fill_value(data['fill_value'], dtype),
            filters=filters,
        )

    def set_for_insertion(self) -> Variable:
        """Create a new variable without any attribute.

        Returns:
            The variable.
        """
        return Variable(name=self.name,
                        dtype=self.dtype,
                        dimensions=self.dimensions,
                        compressor=self.compressor,
                        fill_value=self.fill_value,
                        filters=self.filters)


class Dataset:
    """Handle the metadata of a dataset.

    Args:
        dimensions: A sequence of :py:class:`Dimension` objects representing
            the dimensions of the dataset associated to their size.
        variables: A sequence of :py:class:`Variable` objects representing the
            variables of the dataset.
        attrs: An optional sequence of :py:class:`Attribute` objects
            representing the attributes of the dataset. Defaults to None.
        block_size_limit: An optional integer representing the maximum size
            (in bytes) of a block/chunk of variable's data.
    """
    __slots__ = ('dimensions', 'variables', 'attrs', 'block_size_limit')

    def __init__(self,
                 dimensions: Sequence[Dimension],
                 variables: Sequence[Variable],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 block_size_limit: int | None = None) -> None:
        #: Dimensions of the dataset.
        self.dimensions: dict[str, Dimension] = {
            item.name: item
            for item in dimensions
        }

        #: Variables of the dataset.
        self.variables: dict[str, Variable] = {
            item.name: item
            for item in variables
        }

        #: Attributes of the dataset.
        self.attrs = list(attrs or [])

        #: Maximum data chunk size
        self.block_size_limit: int = block_size_limit or BLOCK_SIZE_LIMIT

    @property
    def dim_chunks(self) -> DimensionType:
        """Dimensions' chunks properties."""
        return {dim.name: dim.chunks for dim in self.dimensions.values()}

    @property
    def dim_size(self) -> DimensionType:
        """Dimensions' size properties."""
        return {dim.name: dim.value for dim in self.dimensions.values()}

    def select_variables(
        self,
        keep_variables: Iterable[str] | None = None,
        drop_variables: Iterable[str] | None = None,
    ) -> set[str]:
        """Select variables to keep or drop from the dataset.

        Args:
            keep_variables: A list of variables to retain from the Dataset.
                If None, all variables are kept.
            drop_variables: A list of variables to exclude from the Dataset.
                If None, no variables are dropped.

        Returns:
            The selected variables.
        """
        result = set(self.variables)

        if keep_variables is not None:
            result &= set(keep_variables)

        if drop_variables is not None:
            result -= set(drop_variables)

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return False
        return self.get_config() == other.get_config()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def get_config(self) -> dict[str, Any]:
        """Get the dataset metadata.

        Returns:
            Dataset configuration.
        """
        attributes = sorted(attr.get_config() for attr in self.attrs)
        variables = tuple(self.variables[name].get_config()
                          for name in sorted(self.variables))

        return {
            'attrs':
            attributes,
            'dimensions':
            tuple(item.get_config() for item in self.dimensions.values()),
            'variables':
            variables,
            'block_size_limit':
            self.block_size_limit,
        }

    def add_dimension(self, dimension: Dimension) -> None:
        """Add a dimension to the dataset.

        Args:
            dimension: dimension to add.

        Raises:
            TypeError: If the dimension is not a Dimension object.
            ValueError: If the variable already exists in the dataset or if
                the variable's dimensions do not match the dataset's
                dimensions.
        """
        if not isinstance(dimension, Dimension):
            raise TypeError(
                f'Dimension must be a Dimension, not {type(dimension)}')
        if dimension.name in self.dimensions:
            raise ValueError(
                f'The dimension {dimension.name!r} already exists in the '
                'collection.')

        self.dimensions[dimension.name] = dimension

    def add_variable(self,
                     variable: Variable,
                     dimensions: set[str] | None = None) -> None:
        """Add a variable to the dataset.

        Args:
            variable: variable to add.
            dimensions: set of known dimensions.

        Raises:
            TypeError: If the variable is not a Variable object.
            ValueError: If the variable already exists in the dataset or if
                the variable's dimensions do not match the dataset's
                dimensions.
        """
        if not isinstance(variable, Variable):
            raise TypeError(
                f'Variable must be a Variable, not {type(variable)}')

        if variable.name in self.variables:
            raise ValueError(
                f'The variable {variable.name!r} already exists in the '
                'collection.')

        dimensions = dimensions or set(self.dimensions)
        # Looking for unknown dimensions.
        if (set(variable.dimensions) | dimensions) != dimensions:
            raise ValueError(
                'The new variable must use the dataset dimensions.')

        self.variables[variable.name] = variable

    @staticmethod
    def from_config(data: dict[str, Any]) -> Dataset:
        """Create a new dataset from the given dataset configuration.

        Args:
            data: dataset configuration.

        Returns:
            New dataset.
        """
        return Dataset(
            dimensions=tuple(
                Dimension.from_config(item)
                for item in data.get('dimensions', [])),
            variables=tuple(
                Variable.from_config(item) for item in data['variables']),
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            block_size_limit=data.get('block_size_limit'))

    @staticmethod
    def from_deprecated_config(data: dict[str, Any]) -> Dataset:
        """Create a new dataset from the given deprecated dataset
        configuration.

        Args:
            data: dataset configuration.

        Returns:
            New dataset.
        """
        chunks = dict(data.get('chunks', []))
        return Dataset(
            dimensions=tuple(
                Dimension.from_config((item, -1, chunks.get(item, -1)))
                for item in data['dimensions']),
            variables=tuple(
                Variable.from_config(item) for item in data['variables']),
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            block_size_limit=data.get('block_size_limit'),
        )

    def missing_variables(self, other: Dataset) -> tuple[str, ...]:
        """Finds the variables in the provided dataset that are not in this
        instance.

        Args:
            other: The dataset to compare against.

        Returns:
            A tuple containing the names of the variables that are defined in
            this dataset but not in the provided dataset.

        Raises:
            ValueError: If the provided dataset does not define one or more
                variables that are defined in this dataset.
        """
        this = set(self.variables)
        others = set(other.variables)

        if len(others - this):
            raise ValueError('The reference dataset does not define the '
                             f'{", ".join(others - this)} variables that are '
                             'defined in this dataset.')

        return tuple(this - others)

    def select_variables_by_dims(self,
                                 dims: Sequence[str],
                                 predicate: bool = True) -> set[str]:
        """Select variables that have at least one dimension in the given
        dimensions depending on the predicate.

        Args:
            dims: A sequence of dimensions to select.
            predicate: A boolean value that determines whether to select
                variables that have the given dimensions (True) or variables
                that don't have the given dimensions (False).

        Returns:
            A set of variable names that have the given dimensions (if predicate
            is True) or don't have the given dimensions (if predicate is False).
        """
        if len(dims) == 0:
            return {
                name
                for name, var in self.variables.items()
                if (len(var.dimensions) == 0) == predicate
            }

        set_of_dims = set(dims)
        return {
            name
            for name, var in self.variables.items()
            if bool(set(var.dimensions) & set_of_dims) == predicate
        }
