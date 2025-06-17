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
import logging

import xarray

from .. import collection, dataset, fs_utils, meta, partitioning

if TYPE_CHECKING:
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


def update_deprecated_collection(
        path: str,
        filesystem: fsspec.AbstractFileSystem | str | None = None) -> None:
    """Update deprecated collection's configuration. A backup of the existing
    collection configuration will be kept.

    Args:
        path: The path to the collection.
        filesystem: The filesystem to use for the collection. This is an
            instance of a subclass of :py:class:`fsspec.AbstractFileSystem`.

        Raises:
            ValueError:
                If the provided directory does not contain a valid collection
                configuration file.
    """

    import json

    import zarr

    from ..fs_utils import join_path

    _LOGGER.warning('Updating collection: %r', path)
    filesystem = fs_utils.get_fs(filesystem)

    config = collection.Collection._config(path)

    if not filesystem.exists(config):
        raise ValueError(f'zarr collection not found at path {path!r}')

    with filesystem.open(config) as stream:
        data: dict[str, Any] = json.load(stream)

    if data.get('version', '0') != '0':
        _LOGGER.error('Collection already updated.')
        return

    ds = meta.Dataset.from_deprecated_config(data['dataset'])
    axis = data['axis']
    main_dimension = ds.variables[axis].dimensions[0]

    dimensions_source = _dimensions_source(ds, main_dimension)

    zcol_partitioning = partitioning.get_codecs(data['partitioning'])
    partitions = list(
        zcol_partitioning.list_partitions(fs=filesystem, path=path))

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

        var_shape = zarr.open(filesystem.get_mapper(var_path)).shape

        dim_size = var_shape[dim_index]
        _LOGGER.warning('Assigning size %r to dimension %r.', dim_size,
                        dimension)

        ds.dimensions[dimension].value = dim_size

    missing_dimensions = set(dimensions_source) - set(ds.dimensions)
    if missing_dimensions:
        raise ValueError('Something is wrong, some dimensions are missing: '
                         f'{missing_dimensions}')

    zcol = collection.Collection(axis,
                                 ds,
                                 zcol_partitioning,
                                 path,
                                 mode='w',
                                 filesystem=filesystem)

    config = zcol._config(path)

    config_back = f'{config}.bak'
    _LOGGER.warning('Copying old configuration to: %r', config_back)
    filesystem.copy(config, config_back)

    _LOGGER.warning('Writing new configuration: %r', path)

    zcol._write_config()


def _dimensions_source(ds: meta.Dataset,
                       main_dimension: str) -> dict[str, meta.Variable]:
    """Associate dimension to a variable containing it.

    Args:
        ds: Dataset from which to get variables information.
        main_dimension: Name of the main dimension.

    Returns:
        Dictionary of dimension associated to a variable containing it.
    """
    dimensions_source: dict[str, meta.Variable] = {}
    dimensions = list(ds.dimensions)

    for dimension in dimensions:
        if dimension == main_dimension:
            continue

        for variable in ds.variables.values():
            if dimension in variable.dimensions:
                dimensions_source[dimension] = variable
                break

    return dimensions_source
