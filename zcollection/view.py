# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
View on a reference collection.
===============================
"""
from typing import ClassVar, Iterable, List, Optional, Sequence, Tuple, Union
import dataclasses
import json
import logging

import dask.array
import fsspec
import zarr

from . import collection, dataset, meta, storage, sync, utilities

#: Module logger.
_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ViewReference:
    """Properties of the collection used as reference by a view.

    Args:
        path: Path to the collection.
        filesystem: The file system used to access the reference collection.
    """
    #: Path to the collection.
    path: str
    #: The file system used to access the reference collection.
    fs: fsspec.AbstractFileSystem = utilities.get_fs("file")


def _create_zarr_array(args: Tuple[str, zarr.Group], base_dir: str,
                       fs: fsspec.AbstractFileSystem, template: str,
                       variable: meta.Variable) -> None:
    """Create a Zarr array, with fill_value being used as the default value for
    uninitialized portions of the array.

    Args:
        args: Tuple of (path, zarr.Group).
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to create the Zarr array.
        template: The variable's name is used as a template for the Zarr array
            to determine the shape of the new variable.
        variable: The properties of the variable to create.
    """
    partition, group = args
    data = dask.array.from_zarr(group[template])

    dirname = fs.sep.join((base_dir, partition))
    mapper = fs.get_mapper(fs.sep.join((dirname, variable.name)))
    zarr.full(data.shape,
              chunks=True,
              dtype=variable.dtype,
              compressor=variable.compressor,
              fill_value=variable.fill_value,
              store=mapper,
              overwrite=True,
              filters=variable.filters)
    storage.write_zattrs(dirname, variable, fs)
    fs.invalidate_cache(dirname)


def _drop_zarr_zarr(args: Tuple[str, zarr.Group],
                    base_dir: str,
                    fs: fsspec.AbstractFileSystem,
                    variable: str,
                    ignore_errors: bool = False) -> None:
    """Drop a Zarr array.

    Args:
        args: Tuple of (path, zarr.Group).
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to delete the Zarr array.
        variable: The name of the variable to drop.
        ignore_errors: If True, ignore errors when dropping the array.
    """
    partition, _group = args
    try:
        fs.rm(fs.sep.join((base_dir, partition, variable)), recursive=True)
        fs.invalidate_cache(fs.sep.join((base_dir, partition)))
    # pylint: disable=broad-except
    # We don't want to fail on errors.
    except Exception:
        if not ignore_errors:
            raise
    # pylint: enable=broad-except


def _load_dataset(
    partition: str,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    selected_variables: Optional[Iterable[str]],
    view_ref: ViewReference,
    variables: Sequence[str],
) -> Optional[Tuple[dataset.Dataset, str]]:
    """Load a dataset from a partition stored in the reference collection and
    merge it with the variables defined in this view.

    Args:
        partition: The partition to process.
        base_dir: Base directory of the view.
        fs: The file system used to access the variables in the view.
        selected_variables: The list of variable to retain from the view
            reference.
        view_ref: The properties of the reference collection.
        variables: The variables to retain from the view
    """
    ds = storage.open_zarr_group(
        view_ref.fs.sep.join((view_ref.path, partition)), view_ref.fs,
        selected_variables)
    if ds is None:
        return None

    _ = {
        ds.add_variable(item.metadata(), item.array)
        for item in (
            storage.open_zarr_array(
                zarr.open(  # type: ignore
                    fs.get_mapper(fs.sep.join((base_dir, partition,
                                               variable))),
                    mode="r"),
                variable) for variable in variables)
    }
    return ds, partition


def _assert_variable_handled(reference: meta.Dataset, view: meta.Dataset,
                             variable: str) -> None:
    """Assert that a variable belongs to a view.

    Args:
        reference: The reference dataset.
        view: The view dataset.
        variable: The variable to check.
    """
    if variable in reference.variables:
        raise ValueError(f"Variable {variable} is read-only")
    if variable not in view.variables:
        raise ValueError(f"Variable {variable} does not exist")


class View:
    """View on a reference collection.

    Args:
        base_dir: Path to the directory where the view is stored.
        view_ref: Access properties for the reference view.
        ds: The dataset handled by this view.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.
    """
    #: Configuration filename of the view.
    CONFIG: ClassVar[str] = ".view"

    def __init__(
        self,
        base_dir: str,
        view_ref: ViewReference,
        *,
        ds: Optional[meta.Dataset] = None,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> None:
        #: Path to the directory where the view is stored.
        self.base_dir = base_dir
        #: The file system used to access the view (default local file system).
        self.fs = utilities.get_fs(filesystem)
        #: The reference collection of the view.
        self.view_ref = collection.open_collection(view_ref.path,
                                                   mode="r",
                                                   filesystem=view_ref.fs)
        #: The metadata of the variables handled by the view.
        self.metadata = ds or meta.Dataset(
            self.view_ref.metadata.dimensions, variables=[], attrs=[])
        #: The synchronizer used to synchronize the view.
        self.synchronizer = synchronizer or sync.NoSync()

        if not self.fs.exists(self.base_dir):
            _LOGGER.info("Creating view %s", self)
            self.fs.makedirs(self.base_dir)
            self._write_config()

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"<filesystem={self.fs.__class__.__name__!r}, "
                f"base_dir={self.base_dir!r}>")

    @classmethod
    def _config(cls, base_dir: str, fs: fsspec.AbstractFileSystem) -> str:
        """Returns the configuration path"""
        return fs.sep.join((base_dir, cls.CONFIG))

    def _write_config(self) -> None:
        """Write the configuration file for the view."""
        config = self._config(self.base_dir, self.fs)
        fs = json.loads(self.view_ref.fs.to_json())
        with self.fs.open(config, mode="w") as stream:
            json.dump(
                dict(base_dir=self.base_dir,
                     metadata=self.metadata.get_config(),
                     view_ref=dict(path=self.view_ref.partition_properties.dir,
                                   fs=fs)),
                stream,  # type: ignore
                indent=4)
        self.fs.invalidate_cache(config)

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> "View":
        """Open a View described by a configuration file.

        Args:
            path: The path to the configuration file.
            filesystem: The filesystem to use for the view.
            synchronizer: The synchronizer to use for the view.

        Returns:
            The view.

        Raises:
            ValueError: If the provided directory does not contain a view.
        """
        _LOGGER.info("Opening view %r", path)
        fs = utilities.get_fs(filesystem)
        config = cls._config(path, fs)
        if not fs.exists(config):
            raise ValueError(f"zarr view not found at path {path!r}")
        with fs.open(config) as stream:
            data = json.load(stream)

        view_ref = data["view_ref"]
        return View(data["base_dir"],
                    ViewReference(
                        view_ref["path"],
                        fsspec.AbstractFileSystem.from_json(
                            json.dumps(view_ref["fs"]))),
                    ds=meta.Dataset.from_config(data["metadata"]),
                    filesystem=filesystem,
                    synchronizer=synchronizer)

    def add_variable(self, variable: meta.Variable) -> None:
        """Add a variable to the view

        Args:
            variable: The variable to add

        Raises:
            ValueError: If the variable already exists

        Example:
            >>> view.add_variable(
            ...    zcollection.meta.Variable(
            ...        "temperature",
            ...        "float32", ("time", "lat", "lon"),
            ...        (zcollection.meta.Attribute("units", "degrees Celsius"),
            ...         zcollection.meta.Attribute("long_name", "temperature")),
            ...        fill_value=-9999.0))
        """
        _LOGGER.info("Adding variable %r", variable.name)
        if (variable.name in self.view_ref.metadata.variables
                or variable.name in self.metadata.variables):
            raise ValueError(f"Variable {variable.name} already exists")
        client = utilities.get_client()
        self.metadata.add_variable(variable)
        template = self.view_ref.metadata.search_same_dimensions_as(variable)

        try:
            storage.execute_transaction(
                client, self.synchronizer,
                client.map(
                    _create_zarr_array,
                    tuple(self.view_ref.iterate_on_records(relative=True)),
                    base_dir=self.base_dir,
                    fs=self.fs,
                    template=template.name,
                    variable=variable))
        except Exception:
            storage.execute_transaction(
                client, self.synchronizer,
                client.map(
                    _drop_zarr_zarr,
                    tuple(self.view_ref.iterate_on_records(relative=True)),
                    base_dir=self.base_dir,
                    fs=self.fs,
                    variable=variable.name,
                    ignore_errors=True))
            raise

        self._write_config()

    def drop_variable(
        self,
        varname: str,
    ) -> None:
        """Drop a variable from the view.

        Args:
            varname: The name of the variable to drop.

        Raise:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Example:
            >>> view.drop_variable("temperature")
        """
        _LOGGER.info("Dropping variable %r", varname)
        _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                 varname)
        client = utilities.get_client()

        variable = self.metadata.variables.pop(varname)
        self._write_config()

        storage.execute_transaction(
            client, self.synchronizer,
            client.map(_drop_zarr_zarr,
                       tuple(self.view_ref.iterate_on_records(relative=True)),
                       base_dir=self.base_dir,
                       fs=self.fs,
                       variable=variable.name))

    def load(
        self,
        *,
        filters: collection.PartitionFilter = None,
        selected_variables: Optional[Iterable[str]] = None,
    ) -> Optional[dataset.Dataset]:
        """Load the view.

        Args:
            filters: The predicate used to filter the partitions to drop.
                To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.

        Returns:
            The dataset.

        Example:
            >>> view.load()
            >>> view.load(filters="time == '2020-01-01'")
            >>> view.load(filters=lambda x: x["time"] == "2020-01-01")
        """
        client = utilities.get_client()
        futures = client.map(
            _load_dataset,
            tuple(self.view_ref.partitions(filters=filters, relative=True)),
            base_dir=self.base_dir,
            fs=self.fs,
            selected_variables=self.view_ref.metadata.select_variables(
                selected_variables),
            view_ref=ViewReference(self.view_ref.partition_properties.dir,
                                   self.view_ref.fs),
            variables=self.metadata.select_variables(selected_variables))

        # The load function returns the path to the partitions and the loaded
        # datasets. Only the loaded datasets are retrieved here and filter None
        # values corresponding to empty partitions.
        arrays: List[dataset.Dataset] = list(
            map(
                lambda item: item[0],  # type: ignore
                filter(lambda item: item is not None,
                       client.gather(futures))))  # type: ignore
        if arrays:
            array = arrays.pop(0)
            if arrays:
                array = array.concat(arrays,
                                     self.view_ref.partition_properties.dim)
            return array
        return None

    def update(
        self,
        func: collection.PartitionCallable,
        variable: str,
        /,
        *args,
        filters: collection.PartitionFilter = None,
        selected_variables: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        """Update a variable stored int the view.

        Args:
            func: The function to apply to calculate the new values for the
                target variable.
            variable: The name of the variable to update.
            filters: The predicate used to filter the partitions to drop.
                To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded. Usefull to load only a
                subset of the view.
            args: The positional arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.

        Raises:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Example:
            >>> def temp_celsius_to_kelvin(
            ...     dataset: zcollection.dataset.Dataset,
            ... ) -> numpy.ndarray:
            ...     return dataset["temperature"].values + 273,15
            >>> view.update(update_temperature, "temperature_kelvin")
        """
        _LOGGER.info("Updating variable %r", variable)
        _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                 variable)
        arrays = []

        client = utilities.get_client()
        futures = client.map(
            _load_dataset,
            tuple(self.view_ref.partitions(filters=filters, relative=True)),
            base_dir=self.base_dir,
            fs=self.fs,
            selected_variables=self.view_ref.metadata.select_variables(
                keep_variables=selected_variables),
            view_ref=ViewReference(self.view_ref.partition_properties.dir,
                                   self.view_ref.fs),
            variables=self.metadata.select_variables(selected_variables))

        # We build the list of arguments to pass to the update routine. That is
        # the dataset and the path to the view partition.
        arrays: List[dataset.Dataset] = list(
            map(
                lambda item: (
                    item[0],  # type: ignore
                    self.fs.sep.join(
                        (self.base_dir, item[1]))),  # type: ignore
                filter(lambda item: item is not None,
                       client.gather(futures))))  # type: ignore

        def wrap_function(parameters):
            """Wrap the function to be applied to the dataset."""
            data, partition = parameters

            # Applying function on partition's data
            array = func(data, *args, **kwargs)

            storage.update_zarr_array(
                dirname=self.fs.sep.join((partition, variable)),
                array=array,
                fs=self.fs,
            )

        futures = client.map(wrap_function, arrays)
        storage.execute_transaction(client, self.synchronizer, futures)


def create_view(
    path: str,
    view_ref: ViewReference,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
) -> View:
    """Create a new view.

    Args:
        path: View storage directory.
        view_ref: Access properties for the reference view.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Returns:
        The created view.

    Example:
        >>> view_ref = ViewReference(
        ...     partition_base_dir="/data/mycollection")
        >>> view = create_view("/home/user/myview", view_ref)
    """
    return View(path,
                view_ref,
                ds=None,
                filesystem=filesystem,
                synchronizer=synchronizer)


def open_view(
    path: str,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
) -> View:
    """Open an existing view.

    Args:
        path: View storage directory.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Returns:
        The opened view.

    Example:
        >>> view = open_view("/home/user/myview")
    """
    return View.from_config(path,
                            filesystem=filesystem,
                            synchronizer=synchronizer)
