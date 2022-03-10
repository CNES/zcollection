# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Configuration metadata
======================
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple
import abc

import numcodecs.abc
import numpy
import zarr.codecs
import zarr.meta

from .typing import DTypeLike


class Pair(abc.ABC):
    """Handle pair key/value.

    Args:
        name: name of the key.
        value: value of the key.
    """
    __slots__ = ("name", "value")

    def __init__(self, name: str, value: Any) -> None:
        #: Name of the key.
        self.name = name
        #: Value of the key.
        self.value = self._encode(value)

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
        return f"{self.__class__.__name__}({self.name!r}, {self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pair):
            return False
        return self.get_config() == other.get_config()

    def get_config(self) -> Tuple[str, Any]:
        """Get the key/value pair configuration."""
        return self.name, self.value

    @staticmethod
    @abc.abstractmethod
    def from_config(data: Tuple[str, Any]) -> "Pair":
        """Create a new Pair from the given key/value pair configuration."""
        ...  # pragma: no cover


class Dimension(Pair):
    """Handle the metadata of a dataset dimension.

    Args:
        name: name of the dimension.
        value: value of the dimension.
    """

    @staticmethod
    def from_config(data: Tuple[str, Any]) -> "Dimension":
        """Creates a new instance from its metadata.

        Returns:
            Dimension: a new dimension.
        """
        return Dimension(*data)


class Attribute(Pair):
    """Handle the metadata of a dataset attribute.

    Args:
        name: name of the attribute.
        value: value of the attribute.
    """

    @staticmethod
    def from_config(data: Tuple[str, Any]) -> "Attribute":
        """Create a new instance from its metadata.

        Args:
            data: attribute configuration.

        Returns:
            Attribute: a new attribute.
        """
        return Attribute(*data)


class Variable:
    """Handle the metadata of a dataset variable

    Args:
        name: name of the variable
        dtype: data type of the variable
        dimensions: names of the dimensions of the variable
        attrs: attributes of the variable
        compressor: compression codec for the variable
        fill_value: fill value for the variable
        filters: filters for the variable
    """
    __slots__ = ("attrs", "compressor", "dimensions", "dtype", "fill_value",
                 "filters", "name")

    def __init__(
            self,
            name: str,
            dtype: DTypeLike,
            dimensions: Optional[Sequence[str]] = None,
            attrs: Optional[Sequence[Attribute]] = None,
            compressor: Optional[numcodecs.abc.Codec] = None,
            fill_value: Optional[Any] = None,
            filters: Optional[Sequence[numcodecs.abc.Codec]] = None) -> None:
        attrs = attrs or tuple()

        #: Attributes of the variable.
        self.attrs = tuple(attrs)
        #: Compression codec for the variable.
        self.compressor = compressor
        #: Dimensions of the variable.
        self.dimensions = tuple(dimensions or ())
        #: Data type of the variable.
        self.dtype = numpy.dtype(dtype)
        #: Fill value for the variable.
        self.fill_value = fill_value
        #: Filter codecs for the variable.
        self.filters = filters or tuple()
        #: Variable name.
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.get_config() == other.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Get the variable metadata.

        Returns:
            variable configuration.
        """
        compressor = self.compressor
        compressor = compressor.get_config(
        ) if compressor is not None else None
        filters = tuple(item.get_config()  # type: ignore
                        for item in self.filters)

        return dict(attrs=sorted((item.get_config() for item in self.attrs)),
                    compressor=compressor,
                    dimensions=self.dimensions,
                    dtype=zarr.meta.encode_dtype(self.dtype),
                    fill_value=zarr.meta.encode_fill_value(
                        self.fill_value, self.dtype),
                    filters=filters,
                    name=self.name)

    @staticmethod
    def from_config(data: Dict[str, Any]) -> "Variable":
        """Create a new variable from the given variable configuration.

        Args:
            data: variable configuration.

        Returns:
            new variable.
        """

        def get_codec(codec):
            """Get the codec from its configuration."""
            return zarr.codecs.get_codec(codec) if codec is not None else None

        dtype = zarr.meta.decode_dtype(data["dtype"])
        filters: Sequence[numcodecs.abc.Codec] = tuple(  # type: ignore
            get_codec(item) for item in data["filters"])

        return Variable(
            data["name"],
            dtype,
            data["dimensions"],
            tuple(Attribute.from_config(item) for item in data["attrs"]),
            get_codec(data["compressor"]),
            zarr.meta.decode_fill_value(data["fill_value"], dtype),
            filters,
        )


class Dataset:
    """Handle the metadata of a dataset

    Args:
        dimensions: dimensions of the dataset
        variables: variables of the dataset
        attrs: attributes of the dataset
    """
    __slots__ = ("dimensions", "variables", "attrs")

    def __init__(self,
                 dimensions: Sequence[str],
                 variables: Sequence[Variable],
                 attrs: Optional[Sequence[Attribute]] = None) -> None:
        #: Dimensions of the dataset.
        self.dimensions = tuple(dimensions)

        #: Variables of the dataset.
        self.variables = dict((item.name, item) for item in variables)

        #: Attributes of the dataset.
        self.attrs = attrs or []

    def select_variables(
        self,
        keep_variables: Optional[Iterable[str]] = None,
        drop_variables: Optional[Iterable[str]] = None,
    ) -> Set[str]:
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

    def get_config(self) -> Dict[str, Any]:
        """Get the dataset metadata

        Returns:
            Dataset configuration.
        """
        return dict(attrs=sorted((item.get_config() for item in self.attrs)),
                    dimensions=self.dimensions,
                    variables=tuple(self.variables[name].get_config()
                                    for name in sorted(self.variables)))

    def add_variable(self, variable: Variable) -> None:
        """Add a variable to the dataset.

        Args:
            variable: variable to add.

        Raises:
            ValueError: if the variable already exists or if the variable
                dimensions don't match the dataset dimensions.
        """
        if variable.name in self.variables:
            raise ValueError(
                f"The variable {variable.name!r} already exists in the "
                "collection.")
        if set(variable.dimensions) != set(self.dimensions):
            raise ValueError(
                "The new variable must use the dataset dimensions.")
        self.variables[variable.name] = variable

    @staticmethod
    def from_config(data: Dict[str, Any]) -> "Dataset":
        """Create a new dataset from the given dataset configuration.

        Args:
            data: dataset configuration.

        Returns:
            New dataset.
        """
        return Dataset(
            dimensions=data["dimensions"],
            variables=tuple(
                Variable.from_config(item) for item in data["variables"]),
            attrs=tuple(Attribute.from_config(item) for item in data["attrs"]),
        )

    def search_same_dimensions_as(self, variable: Variable) -> Variable:
        """Searche for a variable in this dataset with the same dimensions as
        the given variable.

        Args:
            variable: The variable used for searching.

        Returns:
            The variable having the same dimensions as the supplied variable.

        Raises:
            ValueError: If no variable with the same dimensions as the given
                variable is found.
        """
        for item in self.variables.values():
            if item.dimensions == variable.dimensions:
                return item
        raise ValueError("No variable using the same dimensions exists.")

    def missing_variables(self, other: "Dataset") -> Sequence[str]:
        """Finds the variables in this dataset that are defined in the given
        dataset but not in this dataset.

        Args:
            other: The dataset to compare against.

        Returns:
            The names of the missing variables in this dataset compared to the
            data dataset.
        """
        this = set(self.variables)
        others = set(other.variables)

        if len(others - this):
            raise ValueError("This dataset contains variables that are "
                             "missing from the given dataset.")
        return tuple(this - others)
