# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
I/O operations
==============
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence
import collections
import json
import logging

import dask.array.core
import dask.base
from dask.delayed import Delayed as dask_Delayed
import dask.distributed
import dask.local
import fsspec
import numcodecs.abc
import numcodecs.blosc
import numpy
import zarr

from . import dataset, meta, sync
from .fs_utils import join_path
from .type_hints import ArrayLike

#: Name of the attribute storing the names of the dimensions of an array.
DIMENSIONS = '_ARRAY_DIMENSIONS'

#: Configuration file that describes the attributes of an array.
ZATTRS = '.zattrs'

#: Configuration file that describes the attributes of a group.
ZGROUP = '.zgroup'

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
    except:  # noqa: E722
        # Before throwing the exception, we wait until all future scheduled
        # ones finished.
        dask.distributed.wait(awaitables)
        dask.distributed.wait(futures)
        raise


def _to_zarr(array: dask.array.core.Array, mapper: fsspec.FSMap, path: str,
             **kwargs) -> None:
    """Write a Dask array to a Zarr dataset.

    Args:
        array: The Dask array to write.
        mapper: The file system mapper.
        path: The path to the Zarr dataset.
        **kwargs: Additional keyword arguments to pass to the `zarr.create`
            function.
    """
    chunks: list[tuple[int, ...]] = [chunk[0] for chunk in array.chunks]
    target: dask.array.core.Array = zarr.create(
        shape=array.shape,
        chunks=chunks,  # type: ignore[arg-type]
        dtype=array.dtype,
        store=mapper,
        path=path,
        overwrite=True,
        write_empty_chunks=False,
        **kwargs)
    dask.array.core.store(array,
                          target,
                          flush=True,
                          lock=False,
                          compute=True,
                          scheduler=dask.local.get_sync,
                          return_stored=False)


def write_zattrs(
    dirname: str,
    variable: meta.Variable | dataset.Variable,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write the attributes of a variable to a Zarr dataset.

    Args:
        dirname: The storage directory of the Zarr dataset.
        variable: The variable to process.
        fs: The file system on which the Zarr dataset is stored.
    """
    attrs = collections.OrderedDict(item.get_config()
                                    for item in variable.attrs)
    attrs[DIMENSIONS] = variable.dimensions
    with fs.open(join_path(dirname, variable.name, ZATTRS),
                 mode='w') as stream:
        json.dump(attrs, stream, indent=2)  # type: ignore[arg-type]


def write_zarr_variable(
    args: tuple[str, dataset.Variable],
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    *,
    block_size_limit: int | None = None,
    chunks: dict[str, int | str] | None = None,
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
    """
    name: str
    variable: dataset.Variable
    kwargs: dict[str, tuple[numcodecs.abc.Codec, ...]]

    name, variable = args
    kwargs = {'filters': variable.filters}
    data: dask.array.core.Array = variable.array if isinstance(
        variable, dataset.DelayedArray) else dask.array.core.from_array(
            variable.array)

    # If the user has not specified a chunk size, we use the default one.
    # Otherwise, we use the user's choice.
    block_size_limit = block_size_limit or meta.BLOCK_SIZE_LIMIT
    var_chunks: dict[int, int | str] = {
        ix: -1
        for ix in range(variable.ndim)
    } if chunks is None else {
        ix: chunks.get(dim, -1)
        for ix, dim in enumerate(variable.dimensions)
    }
    data = data.rechunk(
        var_chunks,  # type: ignore[arg-type]
        block_size_limit=block_size_limit,
    )

    _to_zarr(array=data,
             mapper=fs.get_mapper(dirname),
             path=name,
             compressor=variable.compressor,
             fill_value=variable.fill_value,
             **kwargs)
    write_zattrs(dirname, variable, fs)


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
    """
    attrs = collections.OrderedDict(item.get_config() for item in zds.attrs)
    with fs.open(join_path(dirname, ZATTRS), mode='w') as stream:
        json.dump(attrs, stream, indent=2)  # type: ignore[arg-type]

    with fs.open(join_path(dirname, ZGROUP), mode='w') as stream:
        json.dump(
            {'zarr_format': 2},
            stream,  # type: ignore[arg-type]
            indent=2,
        )
    zarr.consolidate_metadata(fs.get_mapper(dirname))  # type: ignore[arg-type]
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
                iterables: list[tuple[str, Any]] = [
                    (name, client.scatter(variable))
                    for name, variable in zds.variables.items()
                ]
                futures: list[dask.distributed.Future] = client.map(
                    write_zarr_variable,
                    iterables,
                    block_size_limit=zds.block_size_limit,
                    chunks=zds.chunks,
                    dirname=dirname,
                    fs=fs,
                )
                execute_transaction(
                    client,
                    sync.NoSync(),
                    futures,
                    workers=dask.distributed.get_worker().address)
        else:
            tuple(
                map(
                    lambda item: write_zarr_variable(
                        item,
                        dirname,
                        fs,
                        chunks=zds.chunks,
                        block_size_limit=zds.block_size_limit,
                    ), zds.variables.items()))
        _write_meta(zds, dirname, fs)


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
        return dataset.DelayedArray.from_zarr(array, name, DIMENSIONS)
    return dataset.Array.from_zarr(array, name, DIMENSIONS)


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
        and attributes.
    """
    _LOGGER.debug('Opening Zarr group %r', dirname)
    store: zarr.Group = zarr.open_consolidated(  # type: ignore[arg-type]
        fs.get_mapper(dirname), mode='r')
    # Ignore unknown variables to retain.
    selected_variables = set(selected_variables) & set(
        store) if selected_variables is not None else set(store)
    variables: list[dataset.Variable] = [
        open_zarr_array(
            store[name],  # type: ignore[arg-type]
            name,
            delayed=delayed) for name in selected_variables
    ]

    return dataset.Dataset(
        variables=variables,
        attrs=tuple(dataset.Attribute(*item) for item in store.attrs.items()),
        delayed=delayed,
    )


def update_zarr_array(
    dirname: str,
    array: ArrayLike,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Update a Zarr array with new data.

    Args:
        dirname: The directory where the Zarr array is stored.
        array: The new data to write to the array.
        fs: The file system where the Zarr array is stored.

    Notes:
        This function updates the entire Zarr array with the new data. If the
        array is a Dask array, it must be computed before writing to the Zarr.
        If the array is a masked array and the Zarr array has a fill value, the
        masked values are filled with the fill value before writing to the Zarr
        array.
    """
    _LOGGER.debug('Updating Zarr array %r', dirname)
    store: zarr.Array = zarr.open_array(fs.get_mapper(dirname), mode='a')

    if isinstance(array, dask.array.core.Array):
        array = array.compute()

    if isinstance(array,
                  numpy.ma.MaskedArray) and store.fill_value is not None:
        array = array.filled(store.fill_value)

    store[:] = array

    # Invalidate any cached directory information.
    fs.invalidate_cache(dirname)


def del_zarr_array(
    dirname: str,
    name: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Delete a variable from a Zarr dataset.

    Args:
        dirname: The name of the dataset.
        name: The name of the variable.
        fs: The file system that the dataset is stored on.
    """
    _LOGGER.debug('Deleting Zarr array %r', dirname)
    path: str = join_path(dirname, name)
    if fs.exists(path):
        fs.rm(path, recursive=True)
        zarr.consolidate_metadata(
            fs.get_mapper(dirname),  # type: ignore[arg-type]
        )
        # Invalidate any cached directory information.
        fs.invalidate_cache(dirname)


def add_zarr_array(
    dirname: str,
    variable: meta.Variable,
    template: str,
    fs: fsspec.AbstractFileSystem,
    *,
    chunks: dict[str, int | str] | None = None,
) -> None:
    """Add a variable to a Zarr dataset.

    Args:
        dirname: The name of the dataset.
        variable: The variable to add.
        template: The name of the template variable.
        fs: The file system that the dataset is stored on.
        chunks: Chunk size for each dimension. Defaults to None. See
            :func:`zarr.create` for more information.

    Notes:
        This function adds a new variable to an existing Zarr dataset. The new
        variable is created with the same shape as the template variable, and
        with the specified chunk size (if provided). The function also writes
        the variable's attributes to the dataset, and consolidates the
        dataset's metadata.
    """
    _LOGGER.debug('Adding variable %r to Zarr dataset %r', variable.name,
                  dirname)
    shape: tuple[int, ...] = zarr.open(  # type: ignore[arg-type]
        fs.get_mapper(join_path(dirname, template))).shape

    var_chunks: tuple[int | str, ...] = shape if chunks is None else tuple(
        chunks.get(dim, shape[ix])  # type: ignore[misc]
        for ix, dim in enumerate(variable.dimensions))

    store: fsspec.FSMap = fs.get_mapper(join_path(dirname, variable.name))
    zarr.create(
        shape,
        chunks=var_chunks,  # type: ignore[arg-type]
        dtype=variable.dtype,
        compressor=variable.compressor,  # type: ignore[arg-type]
        fill_value=variable.fill_value,  # type: ignore[arg-type]
        store=store,
        filters=variable.filters)
    write_zattrs(dirname, variable, fs)
    zarr.consolidate_metadata(fs.get_mapper(dirname))  # type: ignore[arg-type]
