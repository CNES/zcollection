# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Abstract base class for indexing.
=================================
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, Union
import abc
import functools
import pathlib

import fsspec
import numpy
import pyarrow
import pyarrow.parquet

from .. import collection, dataset
from ..typing import NDArray

#: Scalar data type for the index.
Scalar = Union[int, float, bytes]

#: Index data type.
DType = Union[Scalar, Iterable[Scalar]]

#: Type of associative dictionary used for index queries, which matches a
#: column of the index to the requested values.
QueryDict = Dict[str, DType]


#: pylint: disable=too-few-public-methods
class IndexingCallable(Protocol):
    """Protocol for indexing the partitions of a collection.

    A partition callable is a function that accepts a dataset and
    returns a numpy structured array to be converted to a DataFrame and
    stored in the index. The function is called for each partition of
    the collection to determine the first and last index of the
    partition that contains the value to be indexed.
    """

    def __call__(
        self,
        ds: dataset.Dataset,
        *args,
        **kwargs,
    ) -> NDArray:
        """Indexing the partition of the collection.

        Args:
            ds: Dataset to be indexed.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A numpy structured array to be converted to a DataFrame and stored
            in the index.
        """
        # pylint: disable=unnecessary-ellipsis
        # Ellipsis is necessary to make the function signature match the
        # protocol.
        ...  # pragma: no cover
        # pylint: enable=unnecessary-ellipsis


class Indexer:
    """Abstract base class for indexing a collection.

    This class defines the interface for indexing a collection.

    Args:
        path: The path to the index.
        filesystem: The filesystem to use.
    """
    #: The name of the column containing the start of the slice.
    START = 'start'

    #: The name of the column containing the stop of the slice.
    STOP = 'stop'

    def __init__(
        self,
        path: pathlib.Path | str,
        *,
        filesystem: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        if isinstance(path, pathlib.Path):
            path = str(path)
        #: Path to the index.
        self._path = path
        #: Filesystem to use.
        self._fs = filesystem or fsspec.filesystem('file')
        #: Metadata to attach to the index.
        self._meta: dict[str, bytes] = {}
        #: Partitioning keys of the indexed collection.
        self._partition_keys: tuple[str, ...] = ()
        #: PyArrow table containing the index.
        self._table: pyarrow.Table | None = None
        #: Type for each columns of the index.
        self._type: dict[str, pyarrow.DataType] = {}

    @property
    def meta(self) -> dict[str, bytes]:
        """Metadata attached to the index.

        Returns:
            The metadata.
        """
        return self._meta

    @classmethod
    def dtype(cls, **_kwargs) -> list[tuple[str, str]]:
        """Return the columns of the index.

        Args:
            **kwargs: Additional arguments to pass to the function.

        Returns:
            A tuple of (name, type) pairs.
        """
        return [
            (cls.START, 'int64'),
            (cls.STOP, 'int64'),
        ]

    @classmethod
    def pyarrow_type(cls, **kwargs) -> dict[str, pyarrow.DataType]:
        """Return the PyArrow DataType for the index.

        Args:
            **kwargs: Additional arguments to pass to the function.

        Returns:
            The PyArrow type.
        """
        dtype = dict(cls.dtype(**kwargs))
        binary = {}
        for name, value in tuple(dtype.items()):
            if value.startswith('S'):
                binary[name] = pyarrow.binary(int(value[1:]))
                del dtype[name]
        result = {
            name: getattr(pyarrow, value)()
            for name, value in dtype.items()
        }
        result.update(binary)
        return result

    def _set_schema(
        self,
        partition_schema: tuple[tuple[str, pyarrow.DataType], ...],
        **kwargs,
    ) -> None:
        """Set the schema properties of the index.

        Args:
            partition_schema: A tuple of (name, type) pairs that describes the
                storage properties of the collection's partitioning keys.
        """
        dtype = self.pyarrow_type(**kwargs)
        self._partition_keys = tuple(item[0] for item in partition_schema)
        self._type = {name: dtype[name] for name, _ in self.dtype()}
        self._type.update({item[0]: item[1] for item in partition_schema})

    def _sort_keys(self) -> list[tuple[str, str]]:
        """Return the list of keys to sort the index by."""
        keys = self._partition_keys + (self.START, self.STOP)
        return [(key, 'ascending') for key in keys]

    @classmethod
    def _create(
        cls,
        path: pathlib.Path | str,
        ds: collection.Collection,
        meta: dict[str, bytes] | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        **kwargs,
    ) -> Indexer:
        """Create a new index.

        Args:
            path: The path to the index.
            ds: The collection to index.
            meta: Metadata to attach to the index.
            filesystem: The filesystem to use.

        Returns:
            The created index.
        """
        partition_schema = tuple((name, getattr(pyarrow, value)())
                                 for name, value in ds.partitioning.dtype())
        self = cls(path, filesystem=filesystem)
        self._meta = meta or {}
        self._set_schema(partition_schema, **kwargs)
        return self

    @classmethod
    @abc.abstractmethod
    def create(
        cls,
        path: pathlib.Path | str,
        ds: collection.Collection,
        *,
        filesystem: fsspec.AbstractFileSystem | None = None,
        **kwargs,
    ) -> Indexer:
        """Create a new index.

        Args:
            path: The path to the index.
            ds: The collection to index.
            filesystem: The filesystem to use.

        Returns:
            The created index.
        """

    @classmethod
    def open(
        cls,
        path: pathlib.Path | str,
        *,
        filesystem: fsspec.AbstractFileSystem | None = None,
    ) -> Indexer:
        """Open an index.

        Args:
            path: The path to the index.
            filesystem: The filesystem to use.

        Returns:
            The index.
        """
        self = cls(path, filesystem=filesystem)
        with self._fs.open(path, 'rb') as stream:
            schema = pyarrow.parquet.read_schema(stream)
        columns = tuple(name for name, _ in self.dtype())
        self._partition_keys = tuple(name for name in schema.names
                                     if name not in columns)
        self._type = {name: schema.field(name).type for name in schema.names}
        self._meta = {
            name.decode(): value
            for name, value in schema.metadata.items()
        } if schema.metadata is not None else {}
        return self

    def _update(
        self,
        ds: collection.Collection,
        func: IndexingCallable,
        partition_size: int | None = None,
        npartitions: int | None = None,
        **kwargs,
    ) -> None:
        """Update the index.

        Args:
            ds: The dataset containing the new data.
            func: The function to use to calculate the index.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            **kwargs: Additional arguments to pass to the function.
        """
        tables: list[pyarrow.Table] = []
        bag = ds.map(func,
                     partition_size=partition_size,
                     npartitions=npartitions,
                     **kwargs)
        # List of new partitions indexed.
        partitions = []
        for partition, data in bag.compute():
            length = data.size
            # If the current item is empty, skip it.
            if length == 0:
                continue
            # Create a new table with the indexed data.
            data = {
                field: pyarrow.array(data[field], type=self._type[field])
                for field in data.dtype.fields
            }
            # Add the partition to the table.
            data.update(
                (name,
                 pyarrow.nulls(length, type=self._type[name]).fill_null(value))
                for name, value in partition)
            # Memoize the updated partitions.
            partitions.append(tuple(
                (name, value) for name, value in partition))
            tables.append(pyarrow.Table.from_pydict(data))

        # The existing index must be updated?
        if self._fs.exists(self._path):
            table = pyarrow.parquet.read_table(self._path, filesystem=self._fs)

            # pylint: disable=no-member
            # Build the list of mask to select the rows to drop.
            mask = []
            for item in partitions:
                mask.append(
                    functools.reduce(
                        pyarrow.compute.and_,  # type:ignore
                        [
                            pyarrow.compute.is_in(  # type:ignore
                                table[name],
                                value_set=pyarrow.array([value],
                                                        type=self._type[name]))
                            for name, value in item
                        ]))
            mask = functools.reduce(pyarrow.compute.or_, mask)  # type:ignore

            # Inserts the previous index without the updated partitions.
            tables.insert(0, table.filter(
                pyarrow.compute.invert(mask)))  # type:ignore
            # pylint: enable=no-member

        if len(tables) == 0:
            # No new data, nothing to do.
            return

        table = pyarrow.concat_tables(tables)
        if len(self._meta) and table.schema.metadata is None:
            table = table.replace_schema_metadata(self._meta)
        pyarrow.parquet.write_table(table, self._path, filesystem=self._fs)
        self._table = table

    @abc.abstractmethod
    def update(
        self,
        ds: collection.Collection,
        *,
        partition_size: int | None = None,
        npartitions: int | None = None,
    ) -> None:
        """Update the index.

        Args:
            ds: The dataset containing the new data.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
        """

    def _read(self) -> pyarrow.Table:
        """Read the index."""
        if self._table is None:
            self._table = pyarrow.parquet.read_table(self._path,
                                                     filesystem=self._fs)
        return self._table

    def _table_2_indexer(self, table: pyarrow.Table,
                         only_partition_keys: bool) -> collection.Indexer:
        """Convert a table to an indexer.

        Args:
            table: The table to convert.
            only_partition_keys: If True, only the partition keys are kept.

        Returns:
            The indexer.
        """
        # Columns to keep. START and STOP are always dropped, as they're
        # exported within the slice.
        column_names = (self._partition_keys if only_partition_keys else tuple(
            filter(lambda item: item not in [self.START, self.STOP],
                   self.table.column_names)))

        # Convert columns of partitioning to numpy arrays.
        data = {name: table[name].to_numpy() for name in column_names}

        # Convert columns of the slice definition to numpy arrays.
        start = table[self.START].to_numpy()
        stop = table[self.STOP].to_numpy()

        # Calculate the indexes of each contiguous slice.
        # index = [0,     1, ...,   69,    70, ...,   134]
        # start = [817, 823, ..., 2320, 17333, ..., 19337]
        # stop  = [832, 874, ..., 2396, 17420, ..., 19389]
        # chunks = [0, 70, 134]
        chunks: Any = [[0],
                       numpy.where(start[1:] - stop[:-1] > 0)[0] + 1,
                       [len(table)]]

        # Adds the chunks corresponding to the partitioning keys
        chunks += [
            numpy.where(numpy.roll(values, 1) != values)[0]
            for values in data.values()
        ]

        # Finally, we build the list of indexes of the different chunks found.
        chunks = numpy.unique(numpy.concatenate(chunks))

        return (
            tuple(  # type:ignore
                (tuple(
                    (name, data[name][ix0])
                    for name in column_names), slice(start[ix0],
                                                     stop[ix1 - 1])), )
            for ix0, ix1 in tuple(zip(chunks[:-1], chunks[1:])))

    def query(
        self,
        columns: QueryDict,
        *,
        logical_op: str | None = None,
        only_partition_keys: bool = True,
    ) -> collection.Indexer:
        """Query the index.

        Args:
            columns: Dictionary of columns to query.
            logical_op: The logical operator to use. Can be "and", "and_not",
                "invert", "or", "xor". Defaults to "and".
            only_partition_keys: If True, only the partition keys are kept.

        Returns:
            Indexer.
        """
        if len(self._partition_keys) == 0:
            return tuple()

        logical_op = logical_op or 'and'
        if logical_op not in ('and', 'and_not', 'invert', 'or', 'xor'):
            raise ValueError(f'Invalid logical operator: {logical_op}')
        if logical_op in ('and', 'or'):
            logical_op += '_'
        function = getattr(pyarrow.compute, logical_op)

        if not set(self._type) & set(columns.keys()):
            raise ValueError(
                f'Invalid column names: {", ".join(columns.keys())}')

        # Transform the columns values into a list if they are not iterable.
        values = {
            k: [v] if not isinstance(v, Iterable) else v
            for k, v in columns.items()
        }

        table = self._read()

        # pylint: disable=no-member
        mask = functools.reduce(function, [
            pyarrow.compute.is_in(table[name],
                                  value_set=pyarrow.array(
                                      value, type=self._type[name]))
            for name, value in values.items()
        ])
        # pylint: disable=no-member
        table = table.filter(mask)

        # The selected table is sorted by the partitioning keys and the slice.
        table = pyarrow.compute.take(
            table,
            pyarrow.compute.sort_indices(table, sort_keys=self._sort_keys()))
        return self._table_2_indexer(table, only_partition_keys)

    @property
    def table(self) -> pyarrow.Table:
        """The index table.

        Returns:
            The index table.

        Raises:
            ValueError: If the index is not initialized.
        """
        if self._fs.exists(self._path):
            return self._read()
        raise ValueError('The index is not initialized.')
