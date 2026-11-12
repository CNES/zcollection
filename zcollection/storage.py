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
    """Execute a transaction in the collection.

    This function executes a transaction in the collection by computing the
    given futures using the provided Dask client. The synchronizer instance is
    used to handle access to critical resources. Any additional keyword
    arguments are passed to the `dask.distributed.compute` function.

    Args:
        client: The Dask client.
        synchronizer: The instance handling access to critical resources.
        futures: Lazy tasks to be done.
        **kwargs: Keyword arguments to pass to `dask.distributed.compute`.

    Returns:
        The result of the transaction.
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
        # Before throwing the exception, we wait until all future scheduled
        # ones finished.
        dask.distributed.wait(awaitables)
        dask.distributed.wait(futures)
        raise


def _to_zarr(array: dask.array.core.Array, dirname: str, name: str,
             fs: fsspec.AbstractFileSystem, **kwargs) -> None:
    """Write a Dask array to a Zarr dataset.

    Args:
        array: The Dask array to write.
        dirname: The storage directory of the Zarr dataset.
        name: The name of the variable to write.
        fs: The file system on which the Zarr dataset is stored.
        **kwargs: Additional keyword arguments to pass to
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
    """Return the attributes an array holds for a variable.

    Args:
        variable: The variable to process.

    Returns:
        Its attributes, and its fill value when it has one.
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

    Args:
        args: The arguments to the function:
            - Name of the variable to write.
            - The variable to write.
        dirname: The target directory.
        fs: The file system on which the Zarr dataset is stored.
        block_size_limit: Maximum size (in bytes) of a block/chunk. Defaults
            to :data:`zcollection.meta.BLOCK_SIZE_LIMIT`.
        chunks: Chunk size for each dimension. Defaults to ``None`` (i.e. the
            default chunk size is used).

    Raises:
        Exception: Whatever the write raises, carrying a note naming the
            variable.
    """
    kwargs: dict[str, tuple[Codec, ...]]

    name, variable = args
    kwargs = {'filters': variable.filters}
    data: dask.array.core.Array = variable.array if isinstance(
        variable, dataset.DelayedArray) else dask.array.core.from_array(
            variable.array)

    # If the user has not specified a chunk size, we use the default one.
    # Otherwise, we use the user's choice.
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
        # If an exception is thrown, we delete the variable from the Zarr
        # dataset.
        del_zarr_array(dirname, name, fs)
        err.add_note(f'while writing the variable {name!r}')
        raise


def _write_meta(
    zds: dataset.Dataset,
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write the metadata of a dataset to a Zarr dataset.

    Args:
        zds: The dataset to process.
        dirname: The storage directory of the Zarr dataset.
        fs: The file system on which the Zarr dataset is stored.

    Raises:
        OSError: If the group does not hold every variable of the dataset.
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
    """Write a partition of a dataset to a Zarr group.

    Args:
        zds: The dataset partition to write.
        dirname: The name of the partition.
        fs: The file system that the partition is stored on.
        synchronizer: The instance handling access to critical resources.
        distributed: Whether to use Dask distributed to write the variables
            in parallel. Defaults to ``True``.

    Writes the variables of the given dataset partition to a Zarr group
    located at the specified directory on the given file system. If
    `distributed` is `True`, the variables are written in parallel using
    Dask distributed. Otherwise, the variables are written sequentially.

    The `synchronizer` argument is an instance of `sync.Sync` that handles
    access to critical resources, such as the Zarr group's metadata and
    attributes. This ensures that multiple processes or threads do not
    attempt to modify the same resource at the same time.
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
    """Write variables from the provided dataset that are missing from the Zarr
    group.

    Args:
        zds: The dataset to write.
        dirname: The name of the partition.
        fs: The file system that the partition is stored on.
        synchronizer: The instance handling access to critical resources.
        distributed: Whether to use Dask distributed to write the variables
            in parallel. Defaults to ``True``.
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
    """Open a Zarr array as a Dask array or a NumPy array.

    Args:
        array: The Zarr array to open.
        name: The name of the variable.
        delayed: Whether to load the variable lazily. If True, returns a Dask
            array. If False, returns a NumPy array. Defaults to True.

    Returns:
        The variable as a Dask array or a NumPy array.
    """
    if delayed:
        return dataset.DelayedArray.from_zarr(array, name)
    return dataset.Array.from_zarr(array, name)


def init_zarr_group(dirname, fs: fsspec.AbstractFileSystem) -> None:
    """Initialize an empty zarr group.

    Args:
        dirname: Group's path.
        fs: The file system that the partition is stored on.
    """
    if fs.exists(dirname):
        return

    _LOGGER.debug('Initializing zarr group: %s', dirname)
    store.create_group(fs, dirname)
    store.consolidate(fs, dirname)


#: What the function :func:`skip_absent_partition` wraps returns.
_T = TypeVar('_T')


class AbsentPartition(FileNotFoundError):
    """A store holds no partition at a path its index names.

    It is raised where a partition is read and nowhere else, so catching
    it catches that one fault and no other file the caller may be
    missing. Its message is the path of the partition. A view reads two
    stores, its reference and its own, and raises this for either.
    """


def read_partition(
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    *,
    delayed: bool = True,
    selected_variables: Iterable[str] | None = None,
) -> dataset.Dataset:
    """Read the partition a path names.

    Args:
        dirname: The path of the partition.
        fs: The file system the partition is held on.
        delayed: Whether to load the data lazily.
        selected_variables: The variables to retain. Defaults to ``None``,
            which retains all of them.

    Returns:
        The dataset the partition holds.

    Raises:
        AbsentPartition: If the store holds no partition at that path.
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
    """Return a function that gives nothing for an absent partition.

    The wrapped function takes the path of a partition as its first
    argument. A call raising :class:`AbsentPartition` is warned about and
    returns ``None``; every other error is raised.

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
    """Warn that the store holds no partition at a path the index names.

    The caller skips that partition. Its index entry remains until
    ``reindex`` or ``validate_partitions(fix=True)`` removes it.

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
    """Open a Zarr group stored in a partition.

    Args:
        dirname: The name of the partition.
        fs: The file system that the partition is stored on.
        delayed: Whether to load the variables lazily. Defaults to True.
        selected_variables: The list of variables to retain from the Zarr
            group. If None, all variables are selected. Defaults to None.

    Returns:
        The Zarr group stored in the partition, with the specified variables
        and attributes. Its variables are ordered by name.
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
    """Return a partition's row count from loaded dimensions or axis metadata.

    Args:
        dimensions: The dimensions of the selected dataset.
        dim: The partitioned dimension.
        axis: The path of the stored axis whose first dimension is dim.
        fs: The file system holding the axis.

    Returns:
        The size of dim. Reading the axis shape loads only its metadata.
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

    If `dirname` contains a group, its metadata is consolidated, whether or not
    the group held the variable.

    Args:
        dirname: The name of the dataset.
        name: The name of the variable.
        fs: The file system that the dataset is stored on.
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
    """Compute the shape of a variable.

    Args:
        variable: Variable for which to compute the shape.
        dimensions: Known dimensions and their size.
        axis: Main axis path.
        fs: The file system that the dataset is stored on.

    Returns:
        Tuple containing the shape of the variable.
    """
    dims = set(variable.dimensions) - set(dimensions)
    if dims:
        # Only a single dimension might be missing: the main axis
        if len(dims) > 1 or axis is None:
            raise ValueError(f'Variable {variable.name} contains'
                             f' unknown dimensions: {dims}.')
        # We do not want to change to original dimensions
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
    """Add a variable to a Zarr dataset.

    Args:
        dirname: The name of the dataset.
        variable: The variable to add.
        dimensions: Known dimensions and their size.
        fs: The file system that the dataset is stored on.
        axis: Axis containing the main dimension data and size.
        chunks: Chunk size for each dimension. Defaults to None. See
            :func:`zcollection.store.create_array` for more information.

    Notes:
        This function adds a new variable to an existing Zarr dataset. The new
        variable is created with the specified chunk size (if provided).
        The function also writes the variable's attributes to the dataset,
        and consolidates the dataset's metadata. A dataset the file system
        does not hold is left alone.
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
) -> bool:
    """Check if a directory contains a valid Zarr group.

    Args:
        dirname: The name of the directory containing the Zarr group to check.
        fs: The file system to use.

    Returns:
        True if the directory contains a valid Zarr group, False otherwise.
    """
    try:
        group: zarr.Group = store.open_group(fs, dirname)
        for _, array in group.arrays():
            data = array[...]
            del data
    except (FileNotFoundError, ValueError, TypeError, RuntimeError):
        return False
    return True
