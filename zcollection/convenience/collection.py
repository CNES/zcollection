# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import json
import logging
import warnings

import xarray

from .. import (
    collection,
    dataset,
    fs_utils,
    meta,
    migration,
    partitioning,
    store,
)
from ..fs_utils import join_path

if TYPE_CHECKING:
    from collections.abc import Iterable

    import fsspec

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


def create_collection(
    axis: str,
    ds: xarray.Dataset | dataset.Dataset,
    partition_handler: partitioning.Partitioning,
    partition_base_dir: str,
    **kwargs,
) -> collection.Collection:
    """Create a collection.

    Args:
        axis: The axis to use for the collection.
        ds: The dataset to use.
        partition_handler: The partitioning handler to use.
        partition_base_dir: The base directory to use for the partitions.
        **kwargs: Additional parameters are passed through to the constructor
            of the class :py:class:`Collection`.

    Example:
        >>> import xarray as xr
        >>> import zcollection
        >>> data = xr.Dataset({
        ...     "a": xr.DataArray([1, 2, 3]),
        ...     "b": xr.DataArray([4, 5, 6])
        ... })
        >>> collection = zcollection.create_collection(
        ...     "a", data,
        ...     zcollection.partitioning.Sequence(("a", )),
        ...     "/tmp/my_collection")

    Returns:
        The collection.

    Raises:
        ValueError: If the base directory already exists.
    """
    filesystem = fs_utils.get_fs(kwargs.pop('filesystem', None))
    if filesystem.exists(partition_base_dir):
        raise ValueError(
            f'The directory {partition_base_dir!r} already exists.')
    if isinstance(ds, xarray.Dataset):
        ds = dataset.Dataset.from_xarray(ds)
    return collection.Collection(axis,
                                 ds.metadata(),
                                 partition_handler,
                                 partition_base_dir,
                                 mode='w',
                                 filesystem=filesystem,
                                 **kwargs)


def open_collection(path: str,
                    *,
                    mode: Literal['r', 'w'] | None = None,
                    **kwargs) -> collection.Collection:
    """Open a collection.

    Args:
        path: The path to the collection.
        mode: The mode to open the collection.
        **kwargs: Additional parameters are passed through the method
            :py:meth:`zcollection.collection.Collection.from_config`.
    Returns:
        The collection.

    Example:
        >>> import zcollection
        >>> collection = zcollection.open_collection(
        ...     "/tmp/mycollection", mode="r")
    """
    return collection.Collection.from_config(path, mode=mode, **kwargs)


#: Start of the warning emitted when a collection written by an earlier
#: release is opened. The functions here open such a collection
#: deliberately.
_DEPRECATED_WARNING = 'This collection was written by an earlier release'


def _open_source(
    path: str,
    fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open the source collection, read only.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.

    Returns:
        The collection.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message=_DEPRECATED_WARNING,
                                category=UserWarning)
        return open_collection(path, mode='r', filesystem=fs)


def _read_config(path: str, fs: fsspec.AbstractFileSystem) -> dict[str, Any]:
    """Read the configuration of a collection.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.

    Returns:
        The configuration.

    Raises:
        ValueError: If the directory holds no collection configuration.
    """
    config: str = collection.Collection._config(path)

    if not fs.exists(config):
        raise ValueError(f'zarr collection not found at path {path!r}')

    with fs.open(config) as stream:
        return json.load(stream)


def _configuration(
    path: str,
    fs: fsspec.AbstractFileSystem,
    data: dict[str, Any],
) -> tuple[str, meta.DatasetSchema, partitioning.Partitioning]:
    """Read the properties a collection is built from.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.
        data: The configuration of the collection.

    Returns:
        The axis, the metadata and the partitioning. A configuration
        written by an earlier release names its dimensions without their
        size: these sizes are read from the first partition.

    Raises:
        ValueError: If the collection holds no partition, or if the
            metadata misses a dimension its variables span.
    """
    zcol_partitioning: partitioning.Partitioning = partitioning.get_codecs(
        data['partitioning'])
    axis: str = data['axis']

    if meta.format_version(data) != meta.DEPRECATED_FORMAT_VERSION:
        return axis, meta.DatasetSchema.from_config(
            data['dataset']), zcol_partitioning

    ds = meta.DatasetSchema.from_deprecated_config(data['dataset'])
    main_dimension = ds.variables[axis].dimensions[0]

    dimensions_source = _dimensions_source(ds, main_dimension)

    partitions = list(zcol_partitioning.list_partitions(fs=fs, path=path))

    if not partitions:
        raise ValueError(
            'The collection requires at least one partition to be updatable.')

    # Using the first partition to extract dimensions information
    partition = partitions[0]

    for dimension, variable in dimensions_source.items():
        dim_index = variable.dimensions.index(dimension)
        if main_dimension in variable.dimensions:
            _LOGGER.warning('Extracting %r size from %r variable.', dimension,
                            variable.name)
            var_path = join_path(partition, variable.name)
        else:
            # This is an immutable variable
            _LOGGER.warning('Extracting %r size from %r constant.', dimension,
                            variable.name)
            var_path = join_path(path, collection.IMMUTABLE, variable.name)

        var_shape = store.open_array(fs, var_path, mode='r').shape

        dim_size = var_shape[dim_index]
        _LOGGER.warning('Assigning size %r to dimension %r.', dim_size,
                        dimension)

        ds.dimensions[dimension].value = dim_size

    missing_dimensions = set(dimensions_source) - set(ds.dimensions)
    if missing_dimensions:
        raise ValueError('Something is wrong, some dimensions are missing: '
                         f'{missing_dimensions}')

    return axis, ds, zcol_partitioning


def _dimensions_source(ds: meta.DatasetSchema,
                       main_dimension: str) -> dict[str, meta.VariableSchema]:
    """Associate dimension to a variable containing it.

    Args:
        ds: Dataset from which to get variables information.
        main_dimension: Name of the main dimension.

    Returns:
        Dictionary of dimension associated to a variable containing it.
    """
    dimensions_source: dict[str, meta.VariableSchema] = {}
    dimensions = list(ds.dimensions)

    for dimension in dimensions:
        if dimension == main_dimension:
            continue

        for variable in ds.variables.values():
            if dimension in variable.dimensions:
                dimensions_source[dimension] = variable
                break

    return dimensions_source


def _matching_target(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open the copy of a collection, read only.

    The axis, the partitioning and the variables of the two are compared,
    which is what tells one collection from another. The size of a
    dimension is not: reading it costs an array of the source, and two
    collections are not told apart by it.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The copy.

    Raises:
        ValueError: If the target is not the copy of the source.
    """
    data = _read_config(source, source_fs)
    zcol = open_collection(target, mode='r', filesystem=target_fs)
    expected = (data['axis'],
                partitioning.get_codecs(data['partitioning']).get_config(),
                {item['name']
                 for item in data['dataset']['variables']})
    if (zcol.axis, zcol.partitioning.get_config(),
            set(zcol.metadata.variables)) != expected:
        raise ValueError(f'The collection {target!r} is not the copy of '
                         f'{source!r}.')
    return zcol


def _open_target(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open the collection a migration writes into.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The target collection. A directory holding no configuration is
        created with the one the source describes.

    Raises:
        ValueError: If the target holds a configuration that disagrees with
            the one the source describes.
    """
    axis, ds, zcol_partitioning = _configuration(
        source, source_fs, _read_config(source, source_fs))

    if not target_fs.exists(collection.Collection._config(target)):
        return collection.Collection(axis,
                                     ds,
                                     zcol_partitioning,
                                     target,
                                     mode='w',
                                     filesystem=target_fs)

    zcol = collection.Collection.from_config(target,
                                             mode='w',
                                             filesystem=target_fs)
    derived = collection.ReadOnlyCollection(axis,
                                            ds,
                                            zcol_partitioning,
                                            target,
                                            mode='r',
                                            filesystem=target_fs)
    if (zcol.axis, zcol.metadata.get_config(),
            zcol.partitioning.get_config()) != (
                derived.axis, derived.metadata.get_config(),
                derived.partitioning.get_config()):
        raise ValueError(
            f'The collection {target!r} is not configured as the one derived '
            f'from {source!r}.')
    return zcol


def migrate(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> migration.Outcome:
    """Copy a collection into a new one, in the layout this release writes.

    The source is read: its partitions, its configuration and its immutable
    variables stay as they are. The first call creates the target, with the
    configuration the source describes.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system to hold the target on.
        filters: The predicate selecting the partitions to copy, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to copy, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions written under ``done``, those the target already
        held complete under ``skipped``, and those the copy stopped on
        under ``failed``, each with the error as text. The immutable
        variables are reported under the name ``.immutable``.

    Raises:
        ValueError: If the target holds a configuration that disagrees with
            the one the source describes.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migrate("/data/old", "/data/new")
        >>> outcome.failed
        ()
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = _open_source(source, source_fs)
    zcol = _open_target(source, target, source_fs, target_fs)

    return migration.migrate_collection(
        src,
        zcol.partition_properties.dir,
        target_fs,
        filters=filters,
        selected_partitions=selected_partitions,
        distributed=distributed)


def status(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
) -> migration.Status:
    """Compare the partitions of a collection with those of the copy it was
    migrated into.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.

    Returns:
        The partitions both hold under ``migrated``, those of the source
        the target does not hold complete under ``missing``, and those the
        target holds alone under ``pruned``. Printing it gives the counts
        of one group per value of the first partitioning key.

    Example:
        >>> import zcollection
        >>> print(zcollection.status("/data/old", "/data/new"))
        year=2014 : 365 migrated, 0 missing
        year=2015 : 212 migrated, 153 missing
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = _open_source(source, source_fs)

    return migration.collection_status(
        src, fs_utils.normalize_path(target_fs, target), target_fs)


def check(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> migration.Outcome:
    """Compare the values of a collection with those of the copy it was
    migrated into.

    Every value of every variable of the selected partitions is read and
    compared, masks included.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to compare, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to compare, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions whose values are equal under ``done``, those the
        target does not hold complete under ``skipped``, and those whose
        variables differ under ``failed``, each with the names of those
        variables.

    Raises:
        ValueError: If one of the two directories holds no collection.
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = _open_source(source, source_fs)
    zcol = _matching_target(source, target, source_fs, target_fs)

    return migration.check_partitions(src,
                                      zcol.partition_properties.dir,
                                      target_fs,
                                      filters=filters,
                                      selected_partitions=selected_partitions,
                                      distributed=distributed)


def prune(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> migration.Outcome:
    """Remove from a collection the partitions its copy holds.

    A partition directory is removed; the files of the collection itself
    stay. Comparing the values of the two collections is the job of
    :py:func:`check <zcollection.check>`.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to remove, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to remove, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions removed under ``done``, those the source no longer
        held under ``skipped``, and those the target does not hold complete
        under ``failed``, each with the reason.

    Raises:
        ValueError: If one of the two directories holds no collection.
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = _open_source(source, source_fs)
    zcol = _matching_target(source, target, source_fs, target_fs)

    return migration.prune_partitions(src,
                                      zcol.partition_properties.dir,
                                      target_fs,
                                      filters=filters,
                                      selected_partitions=selected_partitions,
                                      distributed=distributed)
