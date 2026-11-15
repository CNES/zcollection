# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Abstract base class for indexing.
=================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
import abc
from collections.abc import Iterable
import functools
import itertools
import pathlib

import fsspec
import numpy
import pyarrow
import pyarrow.parquet

if TYPE_CHECKING:
    from .. import collection, dataset
    from ..collection.callable_objects import MapCallable
    from ..type_hints import NDArray

#: Scalar data type for the index.
Scalar = int | float | bytes

#: Index data type.
DType = Scalar | Iterable[Scalar]

#: Mapping of index column names to the values to query.
QueryDict = dict[str, DType]


#: pylint: disable=too-few-public-methods
class IndexingCallable(Protocol):
    """Protocol for functions that index collection partitions.

    An indexing function receives a partition dataset and returns a NumPy
    structured array describing the positions of indexed values. The array
    is converted to a DataFrame and stored in the index.
    """

    def __call__(
        self,
        zds: dataset.Dataset,
        *args,
        **kwargs,
    ) -> NDArray:
        """Build index data for a collection partition.

        Args:
            zds: Dataset to be indexed.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A NumPy structured array to convert to a DataFrame and store
            in the index.
        """

        # Ellipsis is necessary to make the function signature match the
        # protocol.
        ...  # pragma: no cover


class Indexer(abc.ABC):
    """Base class defining the interface for collection indexes.

    Args:
        path: The path to the index.
        filesystem: The file system containing the index.
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
        self._path: str = path
        #: File system containing the index.
        self._fs: fsspec.AbstractFileSystem = \
            filesystem or fsspec.filesystem('file')
        #: Metadata to attach to the index.
        self._meta: dict[str, bytes] = {}
        #: Partitioning keys of the indexed collection.
        self._partition_keys: tuple[str, ...] = ()
        #: PyArrow table containing the index.
        self._table: pyarrow.Table | None = None
        #: Data type of each column of the index.
        self._type: dict[str, pyarrow.DataType] = {}

    @property
    def meta(self) -> dict[str, bytes]:
        """Metadata attached to the index.

        Returns:
            The metadata.
        """
        return self._meta

    def dtype(self, **_kwargs) -> list[tuple[str, str]]:
        """Return the columns of the index.

        Args:
            **_kwargs: Additional arguments to pass to the function.

        Returns:
            A list of ``(name, type)`` pairs.
        """
        return [
            (self.START, 'int64'),
            (self.STOP, 'int64'),
        ]

    def pyarrow_type(self, **kwargs) -> dict[str, pyarrow.DataType]:
        """Return the PyArrow DataType for the index.

        Args:
            **kwargs: Additional arguments to pass to the function.

        Returns:
            The PyArrow type.
        """
        dtype = dict(self.dtype(**kwargs))
        binary: dict[str, pyarrow.DataType] = {}
        for name, value in tuple(dtype.items()):
            if value.startswith('S'):
                binary[name] = pyarrow.binary(int(value[1:]))
                del dtype[name]
        result: dict[str, pyarrow.DataType] = {
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
        """Set the index schema.

        Args:
            partition_schema: A tuple of (name, type) pairs that describes the
                storage properties of the collection's partitioning keys.
        """
        dtype: dict[str, pyarrow.DataType] = self.pyarrow_type(**kwargs)
        self._partition_keys = tuple(item[0] for item in partition_schema)
        self._type = {name: dtype[name] for name, _ in self.dtype()}
        self._type.update({item[0]: item[1] for item in partition_schema})

    def _partition_column(self, name: str, value: Any,
                          length: int) -> tuple[str, pyarrow.Array]:
        """Build a column containing a partition key value.

        Args:
            name: Name of the partitioning variable.
            value: Value the partition has for it.
            length: Number of rows of the column.

        Returns:
            The name of the column and its values.

        Raises:
            ValueError: If the value does not fit the data type the
                partitioning declares for the variable.
        """
        try:
            return name, pyarrow.nulls(length,
                                       type=self._type[name]).fill_null(value)
        except (pyarrow.ArrowInvalid, OverflowError):
            raise ValueError(
                f'The value {value} of {name!r} does not fit the '
                f'{self._type[name]} the partitioning declares for it.'
            ) from None

    def _sort_keys(self) -> list[tuple[str, str]]:
        """Return the keys used to sort the index."""
        keys: tuple[str, ...] = (*self._partition_keys, self.START, self.STOP)
        return [(key, 'ascending') for key in keys]

    @classmethod
    def _create(
        cls,
        path: pathlib.Path | str,
        zds: collection.Collection,
        meta: dict[str, bytes] | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        **kwargs,
    ) -> Indexer:
        """Create a new index.

        Args:
            path: The path to the index.
            zds: The collection to index.
            meta: Metadata to attach to the index.
            filesystem: The file system to use.

        Returns:
            The created index.
        """
        partition_schema = tuple((name, getattr(pyarrow, value)())
                                 for name, value in zds.partitioning.dtype())
        self: Indexer = cls(path, filesystem=filesystem)
        self._meta = meta or {}
        self._set_schema(partition_schema, **kwargs)
        return self

    @classmethod
    @abc.abstractmethod
    def create(
        cls,
        path: pathlib.Path | str,
        zds: collection.Collection,
        *,
        filesystem: fsspec.AbstractFileSystem | None = None,
        **kwargs,
    ) -> Indexer:
        """Create a new index.

        Args:
            path: The path to the index.
            zds: The collection to index.
            filesystem: The file system to use.

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
            filesystem: The file system to use.

        Returns:
            The index.
        """
        self: Indexer = cls(path, filesystem=filesystem)
        with self._fs.open(path, 'rb') as stream:
            schema: pyarrow.Schema = pyarrow.parquet.read_schema(stream)
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
        zcollection: collection.Collection,
        func: MapCallable,
        partition_size: int | None = None,
        npartitions: int | None = None,
        **kwargs,
    ) -> None:
        """Update the index for the selected collection partitions.

        Args:
            zcollection: The collection to index.
            func: The function to use to calculate the index.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            **kwargs: Additional arguments to pass to the function.
        """
        tables: list[pyarrow.Table] = []
        bag = zcollection.map(func,
                              partition_size=partition_size,
                              npartitions=npartitions,
                              **kwargs)
        # Partitions indexed during this update.
        partitions = []
        for partition, data in bag.compute():
            length = data.size
            # If the current item is empty, skip it.
            if length == 0:
                continue
            # Create a new table with the indexed data.
            indexed_data = {
                field: pyarrow.array(data[field], type=self._type[field])
                for field in data.dtype.fields
            }
            # Add the partition to the table.
            indexed_data.update(
                self._partition_column(name, value, length)
                for name, value in partition)
            # Memoize the updated partitions.
            partitions.append(tuple(
                (name, value) for name, value in partition))
            tables.append(pyarrow.Table.from_pydict(indexed_data))

        # If no new data, skip the update.
        if not partitions:
            return

        # Check whether the existing index needs updating.
        if self._fs.exists(self._path):
            table = pyarrow.parquet.read_table(self._path, filesystem=self._fs)

            # Build a mask selecting rows to remove.
            mask = [
                functools.reduce(
                    pyarrow.compute.and_,  # type:ignore[attr-defined]
                    [
                        pyarrow.compute.is_in(  # type:ignore[attr-defined]
                            table[name],
                            value_set=pyarrow.array([value],
                                                    type=self._type[name]))
                        for name, value in item
                    ]) for item in partitions
            ]
            mask = functools.reduce(
                pyarrow.compute.or_,  # type:ignore[attr-defined]
                mask,
            )

            # Keep existing index rows outside the updated partitions.
            tables.insert(0, table.filter(
                pyarrow.compute.invert(mask)))  # type:ignore[attr-defined]

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
        zds: collection.Collection,
        *,
        partition_size: int | None = None,
        npartitions: int | None = None,
    ) -> None:
        """Update the index.

        Args:
            zds: The collection whose partitions are to be indexed.
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

        # Calculate the indices of each contiguous slice.
        # index = [0,     1, ...,   69,    70, ...,   134]
        # start = [817, 823, ..., 2320, 17333, ..., 19337]
        # stop  = [832, 874, ..., 2396, 17420, ..., 19389]
        # chunks = [0, 70, 134]
        chunks: Any = [[0],
                       numpy.where(start[1:] - stop[:-1] > 0)[0] + 1,
                       [len(table)]]

        # Add the chunks corresponding to the partitioning keys.
        chunks += [
            numpy.where(numpy.roll(values, 1) != values)[0]
            for values in data.values()
        ]

        # Build the indices of the resulting chunks.
        chunks = numpy.unique(numpy.concatenate(chunks))

        return (  # force yapf to respect the line break for flake8
            (tuple((name, data[name][ix0])
                   for name in column_names), slice(start[ix0], stop[ix1 - 1]))
            for ix0, ix1 in itertools.pairwise(chunks))

    def query(
        self,
        columns: QueryDict,
        *,
        logical_op: str | None = None,
        mask: pyarrow.ChunkedArray | None = None,
        only_partition_keys: bool = True,
    ) -> collection.Indexer:
        """Query the index for matching column values.

        Args:
            columns: Mapping of column names to the values to match.
            logical_op: The logical operator to use. Can be ``"and"``,
                ``"and_not"``, ``"invert"``, ``"or"`` or ``"xor"``.
                Defaults to ``"and"``.
            mask: An optional mask to apply to the table before querying.
            only_partition_keys: If True, only the partition keys are kept.

        Returns:
            Indexer.
        """
        if len(self._partition_keys) == 0:
            return ()

        logical_op = logical_op or 'and'
        if logical_op not in ('and', 'and_not', 'invert', 'or', 'xor'):
            raise ValueError(f'Invalid logical operator: {logical_op}')
        if logical_op in ('and', 'or'):
            logical_op += '_'
        function = getattr(pyarrow.compute, logical_op)

        if columns and (not set(self._type) & set(columns.keys())):
            raise ValueError(
                f'Invalid column names: {", ".join(columns.keys())}')

        # Wrap non-iterable column values in lists.
        values = {
            k: [v] if not isinstance(v, Iterable) else v
            for k, v in columns.items()
        }

        table = self._read()

        if mask is not None:
            table = table.filter(mask)

        if values:
            mask = functools.reduce(function, [
                pyarrow.compute.is_in(table[name],
                                      value_set=pyarrow.array(
                                          value, type=self._type[name]))
                for name, value in values.items()
            ])

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
