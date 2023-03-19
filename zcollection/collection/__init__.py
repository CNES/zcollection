# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import datetime
import functools
import io
import itertools
import json
import logging
import os
import pathlib
import types

import dask.bag.core
import dask.distributed
import dask.utils
import fsspec
import numpy
import xarray
import zarr
import zarr.storage

from .. import (
    dask_utils,
    dataset,
    expression,
    fs_utils,
    merging,
    meta,
    partitioning,
    storage,
    sync,
)
from .callable_objects import MapCallable, PartitionCallable, UpdateCallable
from .detail import (
    PartitioningProperties,
    _insert,
    _load_and_apply_indexer,
    _load_dataset,
    _load_dataset_with_overlap,
    _wrap_update_func,
    _wrap_update_func_with_overlap,
    try_infer_callable,
)

#: Type of functions filtering the partitions.
PartitionFilterCallback = Callable[[Dict[str, int]], bool]

#: Type of argument to filter the partitions.
PartitionFilter = Optional[Union[str, PartitionFilterCallback]]

#: Indexer's type.
Indexer = Iterable[Tuple[Tuple[Tuple[str, int], ...], slice]]

#: Name of the directory storing the immutable dataset.
_IMMUTABLE = '.immutable'

#: Module logger.
_LOGGER = logging.getLogger(__name__)


def variables(
    metadata: meta.Dataset,
    selected_variables: Iterable[str] | None = None
) -> tuple[dataset.Variable, ...]:
    """Return the variables defined in the dataset.

    Args:
        metadata: Metadata dataset containing variables information.
        selected_variables: The variables to return. If None, all the
            variables are returned.

    Returns:
        The variables defined in the dataset.
    """
    selected_variables = selected_variables or metadata.variables.keys()
    return tuple(
        dataset.Variable(
            v.name, numpy.ndarray((0, ) * len(v.dimensions), v.dtype),
            v.dimensions, v.attrs, v.compressor, v.fill_value, v.filters)
        for k, v in metadata.variables.items() if k in selected_variables)


def build_indexer_args(
    collection: Collection,
    filters: PartitionFilter,
    indexer: Indexer,
    partitions: Iterable[str] | None = None,
) -> Iterator[tuple[tuple[tuple[str, int], ...], list[slice]]]:
    """Build the arguments for the indexer.

    Args:
        collection: The collection to index.
        filters: The partition filters.
        indexer: The indexer.
        partitions: The partitions to index. If None, all the partitions
            are indexed.

    Returns:
        An iterator containing the arguments for the indexer.
    """
    # Build an indexer dictionary between the partition scheme and
    # indexer.
    indexers_map: dict[tuple[tuple[str, int], ...], list[slice]] = {}
    _ = {
        indexers_map.setdefault(  # type: ignore[func-returns-value]
            partition_scheme, []).append(indexer)
        for partition_scheme, indexer in indexer
    }
    # Filter the selected partitions
    partitions = partitions or collection.partitions(filters=filters)
    selected_partitions = set(indexers_map) & {
        collection.partitioning.parse(item)
        for item in partitions
    }

    # For each provided partition scheme, retrieves the corresponding
    # indexer.
    return ((item, indexers_map[item]) for item in sorted(selected_partitions))


def _immutable_path(
    ds: meta.Dataset,
    partition_properties: PartitioningProperties,
) -> str | None:
    """Return the immutable path of the dataset.

    Args:
        ds: The dataset to process.
        partition_properties: The partitioning properties.

    Returns:
        The immutable path of the dataset containing data that are immutable
        relative to the partitioning or None if the dataset does not contain
        immutable data.
    """
    return fs_utils.join_path(
        partition_properties.dir, _IMMUTABLE) if ds.select_variables_by_dims(
            (partition_properties.dim, ), predicate=False) else None


def _write_immutable_dataset(
    ds: dataset.Dataset,
    axis: str,
    path: str,
    fs: fsspec.AbstractFileSystem,
    synchronizer: sync.Sync,
) -> None:
    """Write the immutable dataset.

    Args:
        ds: The dataset to write.
        axis: The partitioning axis.
        path: The path to the immutable dataset.
        fs: The file system that the partition is stored on.
    """
    immutable_dataset = ds.select_variables_by_dims((axis, ), predicate=False)
    assert len(immutable_dataset.variables) != 0, (
        'The dataset to insert does not contain any variable '
        'that is not split.')
    _LOGGER.info('Creating the immutable dataset: %s', path)
    storage.write_zarr_group(immutable_dataset,
                             path,
                             fs,
                             synchronizer,
                             parallel=False)


class Collection:
    """This class manages a collection of files in Zarr format stored in a set
    of subdirectories. These subdirectories split the data, by cycles or dates
    for example, in order to optimize access and updates, deletion or addition
    of new data.

    Args:
        axis: The axis of the collection.
        ds: The dataset containing the collection.
        partition_handler: The partitioning strategy for the collection.
        partition_base_dir: The base directory for the collection.
        mode: The mode of the collection.
        filesystem: The filesystem to use for the collection.
        synchronizer: The synchronizer to use for the collection.

    Raises:
        ValueError: If the axis does not exist in the dataset, if the
            partition key is not defined in the dataset or if the access mode
            is not supported.

    .. note::

        On the first call to the constructor, the dataset given as an argument
        is used to create the metadata of the collection. This information is
        used to validate datasets that are inserted in the collection.

        Normally, this class is not instantiated directly but through the
        :py:meth:`create_collection <zcollection.create_collection>` and
        :py:meth:`open_collection <zcollection.open_collection>` methods of this
        library.
    """
    #: Configuration filename of the collection.
    CONFIG: ClassVar[str] = '.zcollection'

    def __init__(
        self,
        axis: str,
        ds: meta.Dataset,
        partition_handler: partitioning.Partitioning,
        partition_base_dir: str,
        *,
        mode: str | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> None:
        if axis not in ds.variables:
            raise ValueError(
                f'The variable {axis!r} is not defined in the dataset.')

        for varname in partition_handler.variables:
            if varname not in ds.variables:
                raise ValueError(
                    f'The partitioning key {varname!r} is not defined in '
                    'the dataset.')

        mode = mode or 'w'
        if mode not in ('r', 'w'):
            raise ValueError(f'The mode {mode!r} is not supported.')

        #: The axis of the collection.
        self.axis = axis
        #: The metadata that describes the dataset handled by the collection.
        self.metadata = ds
        #: The file system used to read/write the collection.
        self.fs = fs_utils.get_fs(filesystem)
        #: The partitioning strategy used to split the data.
        self.partitioning = partition_handler
        #: The partitioning properties (base directory and dimension).
        self.partition_properties = PartitioningProperties(
            fs_utils.normalize_path(self.fs, partition_base_dir),
            ds.variables[axis].dimensions[0],
        )
        #: The access mode of the collection.
        self.mode = mode
        #: The synchronizer used to synchronize the modifications.
        self.synchronizer = synchronizer or sync.NoSync()
        #: The path to the dataset that contains the immutable data relative
        #: to the partitioning.
        self._immutable = _immutable_path(ds, self.partition_properties)

        self._write_config(skip_if_exists=True)

        if mode == 'r':
            # pylint: disable=method-hidden
            # These methods are overloaded when the collection is opened in
            # readonly.
            for item in [
                    'add_variable',
                    'drop_partitions',
                    'drop_variable',
                    'insert',
                    'update',
            ]:
                assert hasattr(self, item), f'{item} is not a known method.'
                setattr(
                    self, item,
                    types.MethodType(Collection._unsupported_operation, self))
            # pylint: enable=method-hidden

    @property
    def immutable(self) -> bool:
        """Return True if the collection contains immutable data relative to
        the partitioning."""
        return self._immutable is not None

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'<filesystem={self.fs.__class__.__name__!r}, '
                f'partition_base_dir={self.partition_properties.dir!r}, '
                f'mode={self.mode!r}>')

    @staticmethod
    def _unsupported_operation(*args, **kwargs):
        """Raise an exception if the operation is not supported."""
        raise io.UnsupportedOperation('not writable')

    @classmethod
    def _config(cls, partition_base_dir: str) -> str:
        """Return the configuration path."""
        return fs_utils.join_path(partition_base_dir, cls.CONFIG)

    def _write_config(self, skip_if_exists: bool = False) -> None:
        """Write the configuration file."""
        base_dir = self.partition_properties.dir
        config = self._config(base_dir)
        exists = self.fs.exists(config)

        if skip_if_exists and exists:
            return

        message = ("Updating collection's configuration: %s"
                   if exists else 'Creating the collection: %s')
        _LOGGER.info(message, config)

        self.fs.makedirs(base_dir, exist_ok=True)

        params = {
            'axis': self.axis,
            'dataset': self.metadata.get_config(),
            'partitioning': self.partitioning.get_config(),
        }

        with self.fs.open(config, mode='w') as stream:
            json.dump(params, stream, indent=4)  # type: ignore[arg-type]

    def is_readonly(self) -> bool:
        """Return True if the collection is read-only."""
        return self.mode == 'r'

    def is_locked(self) -> bool:
        """Return True if the collection is locked."""
        return self.synchronizer.is_locked()

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        mode: str | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> Collection:
        """Open a Collection described by a configuration file.

        Args:
            path: The path to the configuration file.
            mode: The mode of the collection.
            filesystem: The filesystem to use for the collection.
            synchronizer: The synchronizer to use for the collection.

        Returns:
            The collection.

        Raises:
            ValueError: If the provided directory does not contain a collection.
        """
        _LOGGER.info('Opening collection: %r', path)
        fs = fs_utils.get_fs(filesystem)
        config = cls._config(path)
        if not fs.exists(config):
            raise ValueError(f'zarr collection not found at path {path!r}')
        with fs.open(config) as stream:
            data = json.load(stream)
        return Collection(
            data['axis'],
            meta.Dataset.from_config(data['dataset']),
            partitioning.get_codecs(data['partitioning']),
            path,
            mode=mode or 'r',
            filesystem=fs,
            synchronizer=synchronizer,
        )

    def _is_selected(
        self,
        partition: Sequence[str],
        expr: Callable[[dict[str, int]], bool] | None,
    ) -> bool:
        """Return whether the partition is selected.

        Args:
            partition: The partition to check.
            expr: The expression used to filter the partition.

        Returns:
            Whether the partition is selected.
        """
        if expr is not None:
            return expr(dict(self.partitioning.parse('/'.join(partition))))
        return True

    # pylint: disable=method-hidden
    def insert(
        self,
        ds: xarray.Dataset | dataset.Dataset,
        *,
        merge_callable: merging.MergeCallable | None = None,
        npartitions: int | None = None,
        validate: bool = False,
    ) -> None:
        """Insert a dataset into the collection.

        Args:
            ds: The dataset to insert.
            merge_callable: The function to use to merge the existing data set
                already stored in partitions with the new partitioned data. If
                None, the new partitioned data overwrites the existing
                partitioned data.
            npartitions: The maximum number of partitions to process in
                parallel. By default, partitions are processed one by one.
            validate: Whether to validate dataset metadata before insertion
                or not.

        .. warning::

            Each worker will process a set of independent partitions. However,
            be careful, two different partitions can use the same file (chunk),
            therefore, the library that handles the storage of Dask arrays
            (HDF5, NetCDF, Zarr, etc.) must be compatible with concurrent
            access.

        Raises:
            ValueError: If the dataset mismatched the definition of the
                collection.
        """
        # pylint: disable=method-hidden
        if isinstance(ds, xarray.Dataset):
            ds = dataset.Dataset.from_xarray(ds)

        _LOGGER.info('Inserting of a %s dataset in the collection',
                     dask.utils.format_bytes(ds.nbytes))

        missing_variables = self.metadata.missing_variables(ds.metadata())
        for item in missing_variables:
            variable = self.metadata.variables[item]
            ds.add_variable(variable)

        if validate and not ds.metadata() == self.metadata:
            raise ValueError(
                "Provided dataset's metadata do not match the collection's ones"
            )
        ds = ds.set_for_insertion(ds=self.metadata)
        client = dask_utils.get_client()

        # If the dataset contains variables that should not be partitioned.
        if self._immutable is not None:

            # On the first call, we store the immutable variables in
            # a directory located at the root of the collection.
            if not self.fs.exists(self._immutable):
                _write_immutable_dataset(ds, self.axis, self._immutable,
                                         self.fs, self.synchronizer)

            # Remove the variables that should not be partitioned.
            ds = ds.select_variables_by_dims((self.axis, ))

        # Process the partitions to insert or update by batches to avoid
        # memory issues.
        partitions = tuple(
            self.partitioning.split_dataset(ds, self.partition_properties.dim))

        if npartitions is not None:
            if npartitions < 1:
                raise ValueError('The number of partitions must be positive')
            npartitions = len(partitions) // npartitions + 1

        scattered_ds = client.scatter(ds)
        for sequence in dask_utils.split_sequence(partitions, npartitions):
            futures = [
                dask_utils.simple_delayed('insert', _insert)(
                    partition,  # type: ignore[arg-type]
                    axis=self.axis,
                    ds=scattered_ds,
                    fs=self.fs,
                    merge_callable=merge_callable,
                    partitioning_properties=self.partition_properties,
                ) for partition in sequence
            ]
            storage.execute_transaction(client,
                                        self.synchronizer,
                                        futures,
                                        sync=True)

    def _relative_path(self, path: str) -> str:
        """Return the relative path to the collection.

        Args:
            path: The path to the dataset.

        Returns:
            The relative path to the collection.
        """
        return pathlib.Path(path).relative_to(
            self.partition_properties.dir).as_posix()

    def partitions(
        self,
        *,
        lock: bool = False,
        filters: PartitionFilter = None,
        relative: bool = False,
    ) -> Iterator[str]:
        """List the partitions of the collection.

        Args:
            lock: Whether to lock the collection or not to avoid listing
                partitions while the collection is being modified.
            filters: The predicate used to filter the partitions to load. If
                the predicate is a string, it is a valid python expression to
                filter the partitions, using the partitioning scheme as
                variables. If the predicate is a function, it is a function
                that takes the partition scheme as input and returns a boolean.
            relative: Whether to return the relative path.

        Returns:
            The list of partitions.

        Example:
            >>> tuple(collection.partitions(
            ...     filters="year == 2019 and month == 1"))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02/', ...)
            >>> tuple(collection.partitions(
            ...     filters=lambda x: x["year"] == 2019 and x["month"] == 1))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02/', ...)
        """
        if isinstance(filters, str):
            expr: Any = expression.Expression(filters)
        elif callable(filters):
            expr = filters
        else:
            expr = None

        base_dir = self.partition_properties.dir
        sep = self.fs.sep

        if lock:
            with self.synchronizer:
                partitions: Iterable[str] = tuple(
                    self.partitioning.list_partitions(self.fs, base_dir))
        else:
            partitions = self.partitioning.list_partitions(self.fs, base_dir)

        for item in partitions:
            if item == self._immutable:
                continue
            # Filtering on partition names
            partitions = item.replace(base_dir, '')
            entry = partitions.split(sep)

            if self._is_selected(entry, expr):
                yield self._relative_path(item) if relative else item

    # pylint: disable=method-hidden
    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
        timedelta: datetime.timedelta | None = None,
    ) -> None:
        # pylint: disable=method-hidden
        """Drop the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to drop.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            timedelta: Select the partitions created before the
                specified time delta relative to the current time.

        Example:
            >>> collection.drop_partitions(filters="year == 2019")
            >>> collection.drop_partitions(
            ...     timedelta=datetime.timedelta(days=30))
        """
        now = datetime.datetime.now()
        client = dask_utils.get_client()
        folders = list(self.partitions(filters=filters, lock=True))

        def is_created_before(path: str) -> bool:
            created = self.fs.created(path)
            return now - created > timedelta

        if timedelta is not None:
            folders = list(filter(is_created_before, folders))

        storage.execute_transaction(
            client, self.synchronizer,
            client.map(self.fs.rm, folders, recursive=True))

        def invalidate_cache(path):
            """Invalidate the cache."""
            _LOGGER.info('Dropped partition: %s', path)
            self.fs.invalidate_cache(path)

        tuple(map(invalidate_cache, folders))

    # pylint: disable=duplicate-code
    # false positive, no code duplication
    def map(
        self,
        func: MapCallable,
        *args,
        filters: PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the collection.

        Args:
            func: The function to apply to every partition of the collection.
            *args: The positional arguments to pass to the function.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = collection.map(
            ...     lambda x: (x["var1"] + x["var2"]).values)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
            [5.0, 6.0, 7.0, 8.0]
        """

        def _wrap(
            partition: str,
            func: PartitionCallable,
            selected_variables: Sequence[str] | None,
            *args,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                func: The function to apply.
                partition: The partition to apply the function on.
                selected_variables: The list of variables to retain from the
                    partition.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            ds = _load_dataset(self.fs, self._immutable, partition,
                               selected_variables)
            return self.partitioning.parse(partition), func(
                ds, *args, **kwargs)

        if not callable(func):
            raise TypeError('func must be a callable')

        bag = dask.bag.core.from_sequence(self.partitions(filters=filters),
                                          partition_size=partition_size,
                                          npartitions=npartitions)
        return bag.map(_wrap, func, selected_variables, *args, **kwargs)
        # pylint: enable=duplicate-code

    def map_overlap(
        self,
        func: MapCallable,
        depth: int,
        *args,
        filters: PartitionFilter = None,
        partition_size: int | None = None,
        npartition: int | None = None,
        selected_variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the collection with some
        overlap.

        Args:
            func: The function to apply to every partition of the collection.
                If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition without the overlap.
            depth: The depth of the overlap between the partitions.
            *args: The positional arguments to pass to the function.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            partition_size: The length of each bag partition.
            npartition: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = collection.map_overlap(
            ...     lambda x: (x["var1"] + x["var2"]).values,
            ...     depth=1)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
            [5.0, 6.0, 7.0, 8.0]
        """
        if not callable(func):
            raise TypeError('func must be a callable')

        add_partition_info = dask.utils.has_keyword(func, 'partition_info')

        def _wrap(
            partition: str,
            func: PartitionCallable,
            partitions: tuple[str, ...],
            selected_variables: Sequence[str] | None,
            depth: int,
            *args,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                partition: The partition to apply the function on.
                func: The function to apply.
                partitions: The partitions to apply the function on.
                selected_variables: The list of variables to retain from the
                    partition.
                depth: The depth of the overlap between the partitions.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            ds, indices = _load_dataset_with_overlap(
                depth, self.partition_properties.dim, self.fs, self._immutable,
                partition, partitions, selected_variables)

            if add_partition_info:
                kwargs = kwargs.copy()
                kwargs['partition_info'] = (self.partition_properties.dim,
                                            indices)

            # Finally, apply the function.
            return (self.partitioning.parse(partition),
                    func(ds, *args, **kwargs))

        partitions = tuple(self.partitions(filters=filters))
        bag = dask.bag.core.from_sequence(partitions,
                                          partition_size=partition_size,
                                          npartitions=npartition)
        return bag.map(_wrap, func, partitions, selected_variables, depth,
                       *args, **kwargs)

    def load(
        self,
        *,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
    ) -> dataset.Dataset | None:
        """Load the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to load.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            indexer: The indexer to apply.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.

        Returns:
            The dataset containing the selected partitions.

        .. warning::

            If you select variables to load from the collection, do not insert
            the returned dataset otherwise all skipped variables will be reset
            with fill values.

        Example:
            >>> collection = ...
            >>> collection.load(
            ...     filters="year == 2019 and month == 3 and day % 2 == 0")
            >>> collection.load(
            ...     filters=lambda keys: keys["year"] == 2019 and
            ...     keys["month"] == 3 and keys["day"] % 2 == 0)
        """
        client = dask_utils.get_client()
        arrays: list[dataset.Dataset]
        if indexer is None:
            selected_partitions = tuple(self.partitions(filters=filters))
            if len(selected_partitions) == 0:
                return None

            # No indexer, so the dataset is loaded directly for each
            # selected partition.
            bag = dask.bag.core.from_sequence(
                self.partitions(filters=filters),
                npartitions=dask_utils.dask_workers(client, cores_only=True))
            arrays = bag.map(storage.open_zarr_group,
                             fs=self.fs,
                             selected_variables=selected_variables).compute()
        else:
            # Build the indexer arguments.
            args = tuple(build_indexer_args(self, filters, indexer))
            if len(args) == 0:
                return None

            bag = dask.bag.core.from_sequence(
                args,
                npartitions=dask_utils.dask_workers(client, cores_only=True))

            # Finally, load the selected partitions and apply the indexer.
            arrays = list(
                itertools.chain.from_iterable(
                    bag.map(
                        _load_and_apply_indexer,
                        fs=self.fs,
                        partition_handler=self.partitioning,
                        partition_properties=self.partition_properties,
                        selected_variables=selected_variables,
                    ).compute()))

        array = arrays.pop(0)
        if arrays:
            array = array.concat(arrays, self.partition_properties.dim)
        if self._immutable:
            array.merge(
                storage.open_zarr_group(self._immutable, self.fs,
                                        selected_variables))
        array.fill_attrs(ds=self.metadata)
        return array

    # pylint: disable=method-hidden
    def update(
        self,
        func: UpdateCallable,
        /,
        *args,
        depth: int = 0,
        filters: PartitionFilter | None = None,
        partition_size: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        # pylint: disable=method-hidden
        """Update the selected partitions.

        Args:
            func: The function to apply on each partition.
            *args: The positional arguments to pass to the function.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition.
            filters: The expression used to filter the partitions to update.
            partition_size: The number of partitions to update in a single
                batch. By default, 1 which is the same as to map the function to
                each partition. Otherwise, the function is called on a batch of
                partitions.
            selected_variables: A list of variables to load from the collection.
                If None, all variables are loaded.
            **kwargs: The keyword arguments to pass to the function.

        Example:
            >>> import dask.array
            >>> import zcollection
            >>> def ones(ds):
            ...     return dict(var2=ds.variables["var1"].values * 0 + 1)
            >>> collection = zcollection.Collection("my_collection", mode="w")
            >>> collection.update(ones)
        """
        if not callable(func):
            raise TypeError('func must be a callable')

        try:
            one_partition = next(self.partitions(filters=filters))
        except StopIteration:
            return
        with self.synchronizer:
            ds = storage.open_zarr_group(one_partition, self.fs,
                                         selected_variables)
        func_result = try_infer_callable(func, ds,
                                         self.partition_properties.dim, *args,
                                         **kwargs)
        unknown_variables = set(func_result) - set(
            self.metadata.variables.keys())
        if len(unknown_variables):
            raise ValueError(f'Unknown variables: {unknown_variables}')

        if depth == 0:
            local_func = _wrap_update_func(func, self.fs, self._immutable,
                                           selected_variables, *args, **kwargs)
        else:
            local_func = _wrap_update_func_with_overlap(
                depth, self.partition_properties.dim, func, self.fs,
                self._immutable, selected_variables, *args, **kwargs)

        _LOGGER.info('Updating of the (%s) variable in the collection',
                     ', '.join(repr(item) for item in func_result))
        client = dask_utils.get_client()

        batches = dask_utils.split_sequence(
            tuple(self.partitions(filters=filters, lock=True)), partition_size
            or dask_utils.dask_workers(client, cores_only=True))
        storage.execute_transaction(
            client, self.synchronizer,
            client.map(local_func, tuple(batches), key=func.__name__))

    def _bag_from_partitions(
        self,
        filters: PartitionFilter | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Return a dask bag from the partitions.

        Args:
            filters: The predicate used to filter the partitions to load.
            kwargs: The keyword arguments to pass to the method
                :meth:`partitions`.

        Returns:
            The dask bag.
        """
        partitions = [*self.partitions(filters=filters, **kwargs)]
        return dask.bag.core.from_sequence(seq=partitions,
                                           npartitions=len(partitions))

    def drop_variable(
        self,
        variable: str,
    ) -> None:
        """Delete the variable from the collection.

        Args:
            variable: The variable to delete.

        Raises:
            ValueError: If the variable doesn't exist in the collection or is
                used by the partitioning.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> collection.drop_variable("my_variable")
        """
        _LOGGER.info('Dropping of the %r variable in the collection', variable)
        if variable in self.partitioning.variables:
            raise ValueError(
                f'The variable {variable!r} is part of the partitioning.')
        if variable not in self.metadata.variables:
            raise ValueError(
                f'The variable {variable!r} does not exist in the collection.')
        if self._immutable:
            ds = storage.open_zarr_group(self._immutable, self.fs)
            if variable in ds.variables:
                raise ValueError(
                    f'The variable {variable!r} is part of the immutable '
                    'dataset.')
        client = dask_utils.get_client()
        bag = self._bag_from_partitions(lock=True)
        awaitables = dask.distributed.futures_of(
            bag.map(storage.del_zarr_array, variable, self.fs).persist())
        storage.execute_transaction(client, self.synchronizer, awaitables)
        del self.metadata.variables[variable]
        self._write_config()

    def add_variable(
        self,
        variable: meta.Variable | dataset.Variable,
    ) -> None:
        """Add a variable to the collection.

        Args:
            variable: The variable to add.

        Raises:
            ValueError: if the variable is already part of the collection, it
                doesn't use the partitioning dimension or use a dimension that
                is not part of the dataset.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> new_variable = meta.Variable(
            ...     name="my_variable",
            ...     dtype=numpy.dtype("int16"),
            ...     dimensions=("num_lines", "num_pixels"),
            ...     fill_value=32267,
            ...     attrs=(dataset.Attribute(name="my_attr", value=0), ),
            ... )
            >>> collection.add_variable(new_variable)
        """
        variable = dataset.get_variable_metadata(variable)
        _LOGGER.info('Adding of the %r variable in the collection',
                     variable.name)
        if self.partition_properties.dim not in variable.dimensions:
            raise ValueError(
                'The new variable must use the partitioning axis.')
        self.metadata.add_variable(variable)
        self._write_config()

        client = dask_utils.get_client()

        template = self.metadata.search_same_dimensions_as(variable)
        chunks = {dim.name: dim.value for dim in self.metadata.chunks}
        try:
            bag = self._bag_from_partitions(lock=True)
            futures = dask.distributed.futures_of(
                bag.map(storage.add_zarr_array, variable, template.name,
                        self.fs, chunks).persist())
            storage.execute_transaction(client, self.synchronizer, futures)
        except Exception:
            self.drop_variable(variable.name)
            raise

    def iterate_on_records(
        self,
        *,
        relative: bool = False,
    ) -> Iterator[tuple[str, zarr.Group]]:
        """Iterate over the partitions and the zarr groups.

        Args:
            relative: If True, the paths are relative to the base directory.

        Returns
            The iterator over the partitions and the zarr groups.
        """
        yield from (
            (
                self._relative_path(item) if relative else item,
                zarr.open_consolidated(
                    self.fs.get_mapper(item),  # type: ignore
                    mode='r',
                )) for item in self.partitions())

    def variables(
        self,
        selected_variables: Iterable[str] | None = None
    ) -> tuple[dataset.Variable, ...]:
        """Return the variables of the collection.

        Args:
            selected_variables: The variables to return. If None, all the
                variables are returned.

        Returns:
            The variables of the collection.
        """
        return variables(self.metadata, selected_variables)

    def copy(
        self,
        target: str,
        *,
        filters: PartitionFilter | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        mode: str = 'w',
        npartitions: int | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> Collection:
        """Copy the collection to a new location.

        Args:
            target: The target location.
            filters: The predicate used to filter the partitions to copy.
            filesystem: The file system to use. If None, the file system of the
                collection is used.
            mode: The mode used to open the collection copied. Default is 'w'.
            npartitions: The number of partitions top copy in parallel. Default
                is number of cores.
            synchronizer: The synchronizer used to synchronize the collection
                copied. Default is None.

        Returns:
            The new collection.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="r")
            >>> collection.copy(target="my_new_collection")
        """
        _LOGGER.info('Copying of the collection to %r', target)
        if filesystem is None:
            filesystem = fs_utils.get_fs(target)
        client = dask_utils.get_client()
        npartitions = npartitions or dask_utils.dask_workers(client,
                                                             cores_only=True)

        # Sequence of (source, target) to copy split in npartitions
        args = tuple(
            dask_utils.split_sequence(
                [(item,
                  fs_utils.join_path(
                      target,
                      os.path.relpath(item, self.partition_properties.dir)))
                 for item in self.partitions(filters=filters)], npartitions))
        # Copy the selected partitions
        partial = functools.partial(fs_utils.copy_tree,
                                    fs_source=self.fs,
                                    fs_target=filesystem)

        def worker_task(args: Sequence[tuple[str, str]]) -> None:
            """Function call on each worker to copy the partitions."""
            tuple(map(lambda arg: partial(*arg), args))

        client.gather(client.map(worker_task, args))
        # Then the remaining files in the root directory (config, metadata,
        # etc.)
        fs_utils.copy_files([
            item['name']
            for item in self.fs.listdir(self.partition_properties.dir,
                                        detail=True) if item['type'] == 'file'
        ], target, self.fs, filesystem)
        return Collection.from_config(target,
                                      mode=mode,
                                      filesystem=filesystem,
                                      synchronizer=synchronizer)
