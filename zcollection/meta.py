# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Configuration metadata
======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from . import codecs, serialization

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .type_hints import DTypeLike

#: Block size limit used with dask arrays. (128 MiB)
BLOCK_SIZE_LIMIT = 134217728

#: Type alias for dataset dimensions.
DimensionType = dict[str, int]

#: Version of the configuration layout used for collections and views.
#: Incremented when a format change would prevent older code from reading it.
FORMAT_VERSION = 1

#: Configuration layout version used by earlier package releases.
DEPRECATED_FORMAT_VERSION = 0


def format_version(config: dict[str, Any]) -> int:
    """Read the layout version from a collection or view configuration.

    Args:
        config: The configuration to inspect.

    Returns:
        The layout version.
    """
    version: Any = config.get('version', DEPRECATED_FORMAT_VERSION)
    if isinstance(version, int):
        return version
    # A string identifies the package version that wrote the configuration.
    # ``'0'`` denotes the layout of earlier releases; other strings
    # identify the subsequent layout.
    return (DEPRECATED_FORMAT_VERSION if version == '0' else FORMAT_VERSION)


class Attribute:
    """Metadata for a dataset attribute.

    Args:
        name: The attribute name.
        value: The attribute value.
    """
    __slots__ = ('name', 'value')

    def __init__(self, name: str, value: Any) -> None:
        #: Attribute name.
        self.name: str = name
        #: Attribute value.
        self.value: Any = self._encode(value)

    @staticmethod
    def _encode(value: Any) -> Any:
        """Encode an attribute value for JSON serialization."""
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
        """Return the attribute as a name-value configuration pair."""
        return self.name, self.value

    @staticmethod
    def from_config(data: tuple[str, Any]) -> Attribute:
        """Create an attribute from its configuration.

        Args:
            data: The attribute configuration.

        Returns:
            The new attribute.
        """
        return Attribute(*data)


class Dimension:
    """Metadata for a dataset dimension.

    Args:
        name: The dimension name.
        value: The dimension size, or ``-1`` if unspecified.
        chunks: The chunk size along the dimension.
    """
    __slots__ = ('chunks', 'name', 'value')

    def __init__(self, name: str, value: int, chunks: int = -1) -> None:
        self.name: str = name
        self.value: int = value
        self.chunks: int = chunks

    @staticmethod
    def from_config(data: tuple[str, int, int]) -> Dimension:
        """Create a dimension from its configuration.

        Returns:
            The new dimension.
        """
        return Dimension(*data)

    def get_config(self) -> tuple[str, int, int]:
        """Return the dimension configuration, including its chunk size."""
        return self.name, self.value, self.chunks

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.name!r}, '
                f'{self.value}, {self.chunks})')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return self.get_config() == other.get_config()


class VariableSchema:
    """Metadata defining a dataset variable.

    Args:
        name: The variable name.
        dtype: The variable data type.
        dimensions: The variable dimensions. Defaults to ``None``.
        attrs: The variable attributes. Defaults to ``None``.
        compressor: The compression codec. Defaults to ``None``.
        fill_value: The fill value, used to mask matching values when reading.
            This also applies when the fill value is NaN or NaT. Defaults
            to ``None``.
        filters: The variable filters. Defaults to ``None``.

    Warning:
        When filters are used, ``fill_value`` must be expressed as the
        decoded value. For example, a ``FixedScaleOffset(0, 1000)`` filter
        requires ``65.536`` as the fill value to represent ``65536`` after
        scaling.
    """
    __slots__ = ('attrs', 'compressor', 'dimensions', 'dtype', 'fill_value',
                 'filters', 'name')

    def __init__(self,
                 name: str,
                 dtype: DTypeLike,
                 *,
                 dimensions: Sequence[str] | None = None,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: codecs.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[codecs.Codec] | None = None) -> None:
        attrs = attrs or ()

        #: Attributes of the variable.
        self.attrs = tuple(attrs)
        #: Compression codec for the variable.
        self.compressor: codecs.Codec | None = compressor
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
        if not isinstance(other, VariableSchema):
            return False
        return self.get_config() == other.get_config()

    def get_config(self) -> dict[str, Any]:
        """Return the variable schema configuration.

        Returns:
            The variable configuration.
        """
        compressor_config = codecs.get_config(
            self.compressor) if self.compressor is not None else None

        return {
            'attrs':
            sorted(item.get_config() for item in self.attrs),
            'compressor':
            compressor_config,
            'dimensions':
            self.dimensions,
            'dtype':
            serialization.encode_dtype(self.dtype),
            'fill_value':
            serialization.encode_fill_value(self.fill_value, self.dtype),
            'filters':
            tuple(codecs.get_config(item) for item in self.filters),
            'name':
            self.name,
        }

    @staticmethod
    def from_config(data: dict[str, Any]) -> VariableSchema:
        """Create a variable schema from a configuration.

        Args:
            data: The variable configuration.

        Returns:
            The new variable schema.
        """

        dtype: numpy.dtype = serialization.decode_dtype(data['dtype'])
        filters = tuple(
            codecs.from_config(item) for item in data['filters']
            if item is not None)

        return VariableSchema(
            data['name'],
            dtype,
            dimensions=data['dimensions'],
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            compressor=(codecs.from_config(data['compressor'])
                        if data['compressor'] is not None else None),
            fill_value=serialization.decode_fill_value(data['fill_value'],
                                                       dtype),
            filters=filters,
        )

    def set_for_insertion(self) -> VariableSchema:
        """Return a copy of the variable schema without attributes.

        Returns:
            The new variable schema.
        """
        return VariableSchema(self.name,
                              self.dtype,
                              dimensions=self.dimensions,
                              compressor=self.compressor,
                              fill_value=self.fill_value,
                              filters=self.filters)


class DatasetSchema:
    """Metadata defining a dataset schema.

    Args:
        dimensions: The :py:class:`Dimension` objects describing names, sizes,
            and chunk sizes.
        variables: The :py:class:`VariableSchema` objects for the variables.
        attrs: The dataset :py:class:`Attribute` objects. Defaults to ``None``.
        block_size_limit: The maximum size in bytes of a data chunk.
    """
    __slots__ = ('attrs', 'block_size_limit', 'dimensions', 'variables')

    def __init__(self,
                 dimensions: Sequence[Dimension],
                 variables: Sequence[VariableSchema],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 block_size_limit: int | None = None) -> None:
        #: Dimensions of the dataset.
        self.dimensions: dict[str, Dimension] = {
            item.name: item
            for item in dimensions
        }

        #: Variables of the dataset.
        self.variables: dict[str, VariableSchema] = {
            item.name: item
            for item in variables
        }

        #: Attributes of the dataset.
        self.attrs = list(attrs or [])

        #: Maximum data chunk size.
        self.block_size_limit: int = block_size_limit or BLOCK_SIZE_LIMIT

    @property
    def dim_chunks(self) -> DimensionType:
        """Return a mapping of dimension names to their chunk sizes."""
        return {dim.name: dim.chunks for dim in self.dimensions.values()}

    @property
    def dim_size(self) -> DimensionType:
        """Return a mapping of dimension names to their sizes."""
        return {dim.name: dim.value for dim in self.dimensions.values()}

    def select_variables(
        self,
        keep_variables: Iterable[str] | None = None,
        drop_variables: Iterable[str] | None = None,
    ) -> set[str]:
        """Select variable schemas to retain or remove.

        Args:
            keep_variables: Variables to retain. Defaults to all variables.
            drop_variables: Variables to exclude. Defaults to none.

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
        if not isinstance(other, DatasetSchema):
            return False
        return self.get_config() == other.get_config()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def get_config(self) -> dict[str, Any]:
        """Return the dataset schema configuration.

        Returns:
            The schema configuration.
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
        """Declare a new dimension in the dataset schema.

        Args:
            dimension: The dimension to add.

        Raises:
            TypeError: If the argument is not a ``Dimension``.
            ValueError: If the dimension is already declared.
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
                     variable: VariableSchema,
                     dimensions: set[str] | None = None) -> None:
        """Add a variable to the dataset schema.

        Args:
            variable: The variable to add.
            dimensions: The permitted dimensions. Defaults to those already
                declared in the schema.

        Raises:
            TypeError: If the argument is not a variable schema.
            ValueError: If the variable already exists or uses undeclared
                dimensions.
        """
        if not isinstance(variable, VariableSchema):
            raise TypeError('Expected a VariableSchema, got '
                            f'{type(variable).__name__}')

        if variable.name in self.variables:
            raise ValueError(
                f'The variable {variable.name!r} already exists in the '
                'collection.')

        dimensions = dimensions or set(self.dimensions)
        # Check whether the variable uses undeclared dimensions.
        if (set(variable.dimensions) | dimensions) != dimensions:
            raise ValueError(
                'The new variable must use the dataset dimensions.')

        self.variables[variable.name] = variable

    def drop_variable(self, name: str) -> None:
        """Remove a variable from the dataset schema.

        Args:
            name: The name of the variable to remove.

        Raises:
            ValueError: If the variable does not exist.
        """
        if name not in self.variables:
            raise ValueError(
                f'The variable {name!r} does not exist in the collection.')
        del self.variables[name]

    @staticmethod
    def from_config(data: dict[str, Any]) -> DatasetSchema:
        """Create a dataset schema from its configuration.

        Args:
            data: The dataset configuration.

        Returns:
            The new dataset schema.
        """
        return DatasetSchema(
            tuple(
                Dimension.from_config(item)
                for item in data.get('dimensions', [])),
            tuple(
                VariableSchema.from_config(item)
                for item in data['variables']),
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            block_size_limit=data.get('block_size_limit'))

    @staticmethod
    def from_deprecated_config(data: dict[str, Any]) -> DatasetSchema:
        """Create a dataset schema from an older configuration format.

        Args:
            data: The older dataset configuration.

        Returns:
            The new dataset schema.
        """
        chunks = dict(data.get('chunks', []))
        return DatasetSchema(
            tuple(
                Dimension.from_config((item, -1, chunks.get(item, -1)))
                for item in data['dimensions']),
            tuple(
                VariableSchema.from_config(item)
                for item in data['variables']),
            attrs=tuple(Attribute.from_config(item) for item in data['attrs']),
            block_size_limit=data.get('block_size_limit'),
        )

    def missing_variables(self, other: DatasetSchema) -> tuple[str, ...]:
        """Find variables declared here but missing from another schema.

        Args:
            other: The dataset schema to compare.

        Returns:
            The names of variables declared here but absent from ``other``.

        Raises:
            ValueError: If ``other`` declares variables that are absent from
                this schema.
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
        """Select variable names according to their dimensions.

        Args:
            dims: The dimensions to match. An empty sequence selects the
                variables without dimensions.
            predicate: If ``True``, select variables sharing at least one of
                these dimensions; otherwise, select the other variables.

        Returns:
            The set of selected variable names.
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
