# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
I/O operations
==============
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar
import collections
import functools
import logging

import dask.array.core
import dask.distributed
import dask.local
import numcodecs.blosc
import numpy

from . import dataset, meta, serialization, store, sync
from .fs_utils import join_path
from .variable.abc import FILL_VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from dask.delayed import Delayed as dask_Delayed
    import fsspec
    import zarr

    from .codecs import Codec
    from .type_hints import ArrayLike

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)

#: Disable multithreading in Blosc to avoid competing with Dask.
numcodecs.blosc.use_threads = False


def execute_transaction(
    client: dask.distributed.Client,
    synchronizer: sync.Sync,
    futures: Sequence[dask.distributed.Future | dask_Delayed],
    **kwargs: Any,
) -> Any:
    """Compute a transaction using a Dask client under a synchronizer.

    The synchronizer remains active while the tasks are computed. If a task
    fails, wait for the submitted work to finish before raising the error.

    Args:
        client: The Dask client.
        synchronizer: The synchronizer for the transaction.
        futures: The lazy tasks to compute.
        **kwargs: Additional arguments passed to ``client.compute``.

    Returns:
        The computed results, or ``None`` if no tasks were provided.
    """
    if not futures:
        return None
    awaitables: Iterable[Any] = []
    try:
        with synchronizer:
            awaitables = client.compute(futures,
                                        **kwargs)  # type: ignore[arg-type]
            return client.gather(awaitables)
    except:
        # Wait for all submitted tasks to finish before propagating the error.
        dask.distributed.wait(awaitables)
        dask.distributed.wait(futures)
        raise


def _to_zarr(array: dask.array.core.Array, dirname: str, name: str,
             fs: fsspec.AbstractFileSystem, **kwargs) -> None:
    """Write a Dask array to a Zarr dataset.

    Args:
        array: The Dask array to write.
        dirname: The Zarr dataset directory.
        name: The variable to write.
        fs: The file system containing the Zarr dataset.
        **kwargs: Additional arguments passed to
            :func:`zcollection.store.create_array`.
    """
    _LOGGER.debug('Writing Zarr variable %r', name)

    chunks: list[int] = [chunk[0] for chunk in array.chunks]
    target: zarr.Array = store.create_array(fs,
                                            dirname,
                                            name=name,
                                            shape=array.shape,
                                            chunks=chunks,
                                            dtype=array.dtype,
                                            overwrite=True,
                                            write_empty_chunks=False,
                                            **kwargs)
    dask.array.core.store(
        array,
        # Dask accepts writable Zarr arrays as targets.
        target,  # type: ignore[arg-type]
        flush=True,
        lock=False,
        compute=True,
        scheduler=dask.local.get_sync,
        return_stored=False)


def variable_attributes(
        variable: meta.VariableSchema | dataset.Variable) -> dict[str, Any]:
    """Return the attributes to store for a variable.

    Args:
        variable: The variable to process.

    Returns:
        The variable's attributes, including its fill value if defined.
    """
    attrs = collections.OrderedDict(item.get_config()
                                    for item in variable.attrs)
    if variable.fill_value is not None:
        attrs[FILL_VALUE] = serialization.encode_fill_value(
            variable.fill_value, variable.dtype)
    return attrs


def write_zarr_variable(
    args: tuple[str, dataset.Variable],
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    *,
    block_size_limit: int | None = None,
    chunks: meta.DimensionType | None = None,
) -> None:
    """Write a variable to a Zarr dataset.

    If writing fails, remove the array and raise the original exception with
    an additional note identifying the variable.

    Args:
        args: The variable name and the variable to write.
        dirname: The target directory.
        fs: The target file system.
        block_size_limit: Maximum block size in bytes. Defaults to
            :data:`zcollection.meta.BLOCK_SIZE_LIMIT`.
        chunks: Chunk sizes for each dimension. If ``None``, use the default
            chunk sizes.
    """
    kwargs: dict[str, tuple[Codec, ...]]

    name, variable = args
    kwargs = {'filters': variable.filters}
    data: dask.array.core.Array = variable.array if isinstance(
        variable, dataset.DelayedArray) else dask.array.core.from_array(
            variable.array)

    # Use the configured chunk sizes, or the defaults if none were provided.
    block_size_limit = block_size_limit or meta.BLOCK_SIZE_LIMIT
    var_chunks: dict[int, int | str] = dict.fromkeys(range(
        variable.ndim), -1) if chunks is None else {
            ix: chunks.get(dim, -1)
            for ix, dim in enumerate(variable.dimensions)
        }
    data = data.rechunk(chunks=var_chunks, block_size_limit=block_size_limit)

    try:
        _to_zarr(data,
                 dirname,
                 name,
                 fs,
                 compressor=variable.compressor,
                 fill_value=variable.fill_value,
                 dimensions=variable.dimensions,
                 attributes=variable_attributes(variable),
                 **kwargs)
    except Exception as err:
        # Remove the Zarr array if writing fails.
        del_zarr_array(dirname, name, fs)
        err.add_note(f'while writing the variable {name!r}')
        raise


def _write_meta(
    zds: dataset.Dataset,
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write and consolidate a Zarr group's metadata.

    Args:
        zds: The dataset to process.
        dirname: The Zarr dataset directory.
        fs: The file system containing the dataset.

    Raises:
        OSError: If the group is missing any variable from the dataset.
    """
    attrs = collections.OrderedDict(item.get_config() for item in zds.attrs)
    store.write_group_attributes(fs, dirname, attrs)
    group: zarr.Group = store.consolidate(fs, dirname)
    missing: set[str] = set(zds.variables) - set(group)
    if missing:
        raise OSError(f'{dirname!r} does not hold the variables written to '
                      f'it: {", ".join(sorted(missing))}')
    fs.invalidate_cache(dirname)


def write_zarr_group(
    zds: dataset.Dataset,
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    synchronizer: sync.Sync,
    *,
    distributed: bool = True,
) -> None:
    """Write a dataset partition to a Zarr group.

    The variables are written first, followed by the group attributes and
    consolidated metadata. The synchronizer is used for the entire write.

    Args:
        zds: The dataset partition to write.
        dirname: The partition directory.
        fs: The file system where the partition is stored.
        synchronizer: The synchronizer used for the entire write.
        distributed: Whether to write variables in parallel using Dask.
            If ``False``, variables are written sequentially. Defaults to
            ``True``.
    """
    with synchronizer:
        if distributed:
            with dask.distributed.worker_client() as client:
                # Keep input keys independent of earlier writes being released.
                iterables: list[tuple[str, Any]] = [
                    (name, client.scatter(variable, hash=False))
                    for name, variable in zds.variables.items()
                ]
                futures: list[dask.distributed.Future] = client.map(
                    write_zarr_variable,
                    iterables,
                    chunks=zds.dim_chunks,
                    block_size_limit=zds.block_size_limit,
                    dirname=dirname,
                    fs=fs)
                execute_transaction(
                    client,
                    sync.NoSync(),
                    futures,
                    workers=dask.distributed.get_worker().address)
        else:
            tuple(
                write_zarr_variable(  # type: ignore[func-returns-value]
                    item,
                    dirname,
                    fs,
                    chunks=zds.dim_chunks,
                    block_size_limit=zds.block_size_limit,
                ) for item in zds.variables.items())

        _write_meta(zds, dirname, fs)


def write_zarr_group_missing(
    zds: dataset.Dataset,
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    synchronizer: sync.Sync,
    *,
    distributed: bool = True,
) -> None:
    """Write any dataset variables missing from a Zarr group.

    Args:
        zds: The source dataset.
        dirname: The partition directory.
        fs: The file system containing the partition.
        synchronizer: The synchronizer for the write.
        distributed: Whether to write variables in parallel using Dask.
            Defaults to ``True``.
    """

    _LOGGER.debug('Opening Zarr group %r', dirname)
    group = store.open_group(fs, dirname)
    selected_variables = set(zds.variables) - set(group)

    if not selected_variables:
        return

    zds = zds.select_vars(names=list(selected_variables))
    write_zarr_group(zds, dirname, fs, synchronizer, distributed=distributed)


def open_zarr_array(
    array: zarr.Array,
    name: str,
    *,
    delayed: bool = True,
) -> dataset.Variable:
    """Open a Zarr array as a ZCollection variable.

    Args:
        array: The Zarr array to open.
        name: The variable name.
        delayed: Whether to return a lazily loaded variable. If ``True``,
            return a ``DelayedArray`` backed by Dask; otherwise, return an
            ``Array`` backed by NumPy. Defaults to ``True``.

    Returns:
        The ZCollection variable.
    """
    if delayed:
        return dataset.DelayedArray.from_zarr(array, name)
    return dataset.Array.from_zarr(array, name)


def init_zarr_group(dirname, fs: fsspec.AbstractFileSystem) -> None:
    """Initialize a Zarr group if its directory does not exist.

    Args:
        dirname: The group directory.
        fs: The file system for the group.
    """
    if fs.exists(dirname):
        return

    _LOGGER.debug('Initializing zarr group: %s', dirname)
    store.create_group(fs, dirname)
    store.consolidate(fs, dirname)


#: Return type of functions wrapped by :func:`skip_absent_partition`.
_T = TypeVar('_T')


class AbsentPartition(FileNotFoundError):
    """Raised when a partition referenced by the index is absent from storage.

    The exception is a subclass of ``FileNotFoundError`` and is raised only
    when reading an absent partition. Catching it does not suppress errors
    for other missing files. Its message contains the partition path.

    A view can raise this exception for a partition missing from either
    its own storage or that of its reference collection.
    """


def read_partition(
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    *,
    delayed: bool = True,
    selected_variables: Iterable[str] | None = None,
) -> dataset.Dataset:
    """Read a partition from storage.

    Args:
        dirname: The partition path.
        fs: The file system containing the partition.
        delayed: Whether to load the data lazily.
        selected_variables: Variables to include. If ``None``, include all.

    Returns:
        The partition dataset.

    Raises:
        AbsentPartition: If the partition is absent from storage.
    """
    try:
        return open_zarr_group(dirname,
                               fs,
                               delayed=delayed,
                               selected_variables=selected_variables)
    except FileNotFoundError as err:
        # Distinguish an absent partition from an incomplete one using
        # a fresh directory listing.
        fs.invalidate_cache(dirname)
        if fs.exists(dirname):
            raise
        raise AbsentPartition(dirname) from err


def skip_absent_partition(func: Callable[..., _T]) -> Callable[..., _T | None]:
    """Wrap a function to skip absent partitions.

    The wrapped function takes a partition path as its first argument.
    If it raises :class:`AbsentPartition`, the wrapper logs a warning and
    returns ``None``. Other exceptions propagate.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(partition: str, *args, **kwargs) -> _T | None:
        try:
            return func(partition, *args, **kwargs)
        except AbsentPartition:
            warn_absent_partition(partition)
            return None

    return wrapper


def warn_absent_partition(dirname: str) -> None:
    """Log a warning when an indexed partition is absent from storage.

    The caller skips the partition. Its index entry remains until it is
    removed by ``reindex`` or ``validate_partitions(fix=True)``.

    Args:
        dirname: The path of the absent partition.
    """
    _LOGGER.warning(
        'The store holds no partition at %r, which the index names. Call '
        'reindex to compare the index with the store.', dirname)


def open_zarr_group(
    dirname,
    fs: fsspec.AbstractFileSystem,
    *,
    delayed: bool = True,
    selected_variables: Iterable[str] | None = None,
) -> dataset.Dataset:
    """Read a Zarr group as a ZCollection dataset.

    Args:
        dirname: The partition directory.
        fs: The file system containing the partition.
        delayed: Whether to load variables lazily. Defaults to ``True``.
        selected_variables: Variables to include, or ``None`` for all.
            Defaults to ``None``.

    Returns:
        The dataset with the selected variables and group attributes.
        Variables are ordered by name.
    """
    _LOGGER.debug('Opening Zarr group %r', dirname)
    group = store.open_group(fs, dirname)
    # Ignore unknown variables to retain.
    selected_variables = set(selected_variables) & set(
        group) if selected_variables is not None else set(group)

    variables = [
        open_zarr_array(
            group[name],  # type: ignore[arg-type]
            name,
            delayed=delayed) for name in sorted(selected_variables)
    ]

    return dataset.Dataset(
        variables,
        attrs=tuple(dataset.Attribute(*item) for item in group.attrs.items()),
        delayed=delayed)


def partition_dimension_size(
    *,
    dimensions: meta.DimensionType,
    dim: str,
    axis: str,
    fs: fsspec.AbstractFileSystem,
) -> int:
    """Return the size of a partitioned dimension.

    Args:
        dimensions: The known dimensions of the selected dataset.
        dim: The partitioned dimension.
        axis: The stored axis path, with ``dim`` as its first dimension.
        fs: The file system containing the axis.

    Returns:
        The size of ``dim``. Reading the axis shape requires only metadata.
    """
    if dim in dimensions:
        return dimensions[dim]
    return store.open_array(fs, axis, mode='r').shape[0]


def update_zarr_array(
    dirname: str,
    array: ArrayLike,
    fs: fsspec.AbstractFileSystem,
    *,
    selection: tuple[Any, ...] | None = None,
) -> None:
    """Update a Zarr array with new data.

    Args:
        dirname: The directory where the Zarr array is stored.
        array: The new data to write to the array.
        fs: The file system where the Zarr array is stored.
        selection: Orthogonal selection of the elements to write, one entry
            per dimension. Defaults to the whole array.

    Notes:
        A Dask array is computed before it is written. When the Zarr array
        has a fill value, it replaces the masked values of a masked array.
    """
    _LOGGER.debug('Updating Zarr array %r', dirname)
    target: zarr.Array = store.open_array(fs, dirname, mode='r+')

    if isinstance(array, dask.array.core.Array):
        array = array.compute()

    if isinstance(array,
                  numpy.ma.MaskedArray) and target.fill_value is not None:
        array = array.filled(target.fill_value)

    if selection is None:
        target[:] = array
    else:
        target.set_orthogonal_selection(selection, array)

    # Invalidate any cached directory information.
    fs.invalidate_cache(dirname)


def del_zarr_array(
    dirname: str,
    name: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Delete a variable from a Zarr dataset.

    If ``dirname`` contains a group, consolidate its metadata even if the
    variable was already absent.

    Args:
        dirname: The dataset directory.
        name: The variable to delete.
        fs: The file system containing the dataset.
    """
    _LOGGER.debug('Deleting Zarr array %r', dirname)
    path = join_path(dirname, name)
    if fs.exists(path):
        fs.rm(path, recursive=True)
    # A previous call may have deleted the array and failed to consolidate.
    if store.contains_group(fs, dirname):
        store.consolidate(fs, dirname)
    fs.invalidate_cache(dirname)


def variable_shape(
    *,
    variable: meta.VariableSchema,
    dimensions: meta.DimensionType,
    axis: str | None,
    fs: fsspec.AbstractFileSystem,
) -> tuple[int, ...]:
    """Compute the shape of a variable from known dimensions.

    Args:
        variable: The variable to size.
        dimensions: Known dimension names and sizes.
        axis: The stored axis path, used to determine one missing size.
        fs: The file system containing the dataset.

    Returns:
        The shape of the variable.
    """
    dims = set(variable.dimensions) - set(dimensions)
    if dims:
        # At most one dimension can be missing: the partitioned dimension.
        if len(dims) > 1 or axis is None:
            raise ValueError(f'Variable {variable.name} contains'
                             f' unknown dimensions: {dims}.')
        # Preserve the original dimensions.
        dimensions = dimensions.copy()
        dim_main, = dims
        dimensions[dim_main] = store.open_array(fs, axis, mode='r').size

    return tuple(dimensions[x] for x in variable.dimensions)


def add_zarr_array(
    dirname: str,
    variable: meta.VariableSchema,
    dimensions: meta.DimensionType,
    fs: fsspec.AbstractFileSystem,
    *,
    axis: str | None = None,
    chunks: dict[str, int] | None = None,
) -> None:
    """Add a variable to an existing Zarr dataset.

    If the dataset directory is missing, log a warning and leave it unchanged.
    The new array receives the variable's attributes, and the group's
    metadata is consolidated.

    Args:
        dirname: The dataset directory.
        variable: The variable to add.
        dimensions: Known dimension names and sizes.
        fs: The file system containing the dataset.
        axis: The stored axis path used to obtain the partitioned size.
        chunks: Chunk sizes for each dimension, or ``None`` for defaults.
            See :func:`zcollection.store.create_array`.
    """
    _LOGGER.debug('Adding variable %r to Zarr dataset %r', variable.name,
                  dirname)
    if not fs.exists(dirname):
        _LOGGER.warning(
            'The variable %r is not added to %r, which the file system does '
            'not hold. Reindex the collection to compare its index with its '
            'store.', variable.name, dirname)
        return

    var_shape = variable_shape(
        variable=variable,
        dimensions=dimensions,
        axis=join_path(dirname, axis) if axis is not None else None,
        fs=fs)

    var_chunks: tuple[int, ...] = var_shape if chunks is None else tuple(
        chunks.get(dim, var_shape[ix])
        for ix, dim in enumerate(variable.dimensions))

    store.create_array(fs,
                       join_path(dirname, variable.name),
                       shape=var_shape,
                       chunks=var_chunks,
                       dtype=variable.dtype,
                       compressor=variable.compressor,
                       fill_value=variable.fill_value,
                       filters=variable.filters,
                       dimensions=variable.dimensions,
                       attributes=variable_attributes(variable))
    store.consolidate(fs, dirname)


def check_zarr_group(
        dirname: str,
        fs: fsspec.AbstractFileSystem,
        variables: Iterable[str] = (),
) -> bool:
    """Check whether a directory contains a readable Zarr group.

    Treat missing files and known reading errors as validation failures;
    other exceptions propagate.

    Args:
        dirname: The directory to check.
        fs: The file system containing the directory.
        variables: The arrays that must be present in the group.

    Returns:
        ``True`` if the group opens, contains the required arrays, and all
        arrays can be read; ``False`` if a handled validation error occurs.
    """
    required = set(variables)
    try:
        group: zarr.Group = store.open_group(fs, dirname)
        for name in required:
            data = store.open_array(fs, join_path(dirname, name),
                                    mode='r')[...]
            del data
        for name, array in group.arrays():
            if name not in required:
                data = array[...]
                del data
    except (FileNotFoundError, ValueError, TypeError, RuntimeError):
        return False
    return True
