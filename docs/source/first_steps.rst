First Steps
===========

This section outlines the steps required to get started with the main features
of the library. Before starting, make sure the library is configured to run on
your machine.

Initialization of the environment
---------------------------------

Before starting, we will create a dataset to handle our collection. ::

    >>> import zcollection.tests.data
    >>> def create_dataset():
    >>>     generator = zcollection.tests.data.create_test_dataset_with_fillvalue()
    >>>     return next(generator)
    >>> ds = create_dataset()
    >>> ds.to_xarray()
    <xarray.Dataset>
    Dimensions:  (num_lines: 61, num_pixels: 25)
    Dimensions without coordinates: num_lines, num_pixels
    Data variables:
        time     (num_lines) datetime64[ns] dask.array<chunksize=(61,), meta=np.ndarray>
        var1     (num_lines, num_pixels) float64 dask.array<chunksize=(61, 25), meta=np.ndarray>
        var2     (num_lines, num_pixels) float64 dask.array<chunksize=(61, 25), meta=np.ndarray>
    Attributes:
        attr:     1

Then we will create a file system in memory. ::

    >>> import fsspec
    >>> fs = fsspec.filesystem('memory')

Finally we create a local dask cluster using only threads in order to work with
the file system stored in memory. ::

    >>> import dask.distributed
    >>> cluster = dask.distributed.LocalCluster(processes=False)
    >>> client = dask.distributed.Client(cluster)

Collection
----------

Creation of a collection
^^^^^^^^^^^^^^^^^^^^^^^^

This introduction will describe the main functionalities allowing to handle a
collection : create, open, load, modify a collection.

Before creating our collection, we define the partitioning of our dataset. In
this example, we will partition the data by ``month`` using the variable
``time``. ::

    >>> import zcollection
    >>> partition_handler = zcollection.partitioning.Date(
    >>>     ("time", ), resolution="M")

Finally, we create our collection: ::

    >>> collection = zcollection.create_collection("time",
    >>>                                             ds,
    >>>                                             partition_handler,
    >>>                                             "/my_collection",
    >>>                                             filesystem=fs)

.. note::

    The collection created can be accessed using the following command ::

        >>> collection = zcollection.open_collection("/my_collection",
        >>>                                          filesystem=fs)

When the collection has been created, a configuration file is created. This file
contains all the metadata to ensure that all future inserted data will have the
same features as the existing data (data consistency). ::

    >>> collection.metadata.get_config()
    {'attrs': [('attr', 1)],
     'dimensions': ('num_lines', 'num_pixels'),
     'variables': ({'attrs': [('attr', 1)],
       'compressor': {'id': 'blosc',
        'cname': 'lz4',
        'clevel': 5,
        'shuffle': 1,
        'blocksize': 0},
       'dimensions': ('num_lines',),
       'dtype': '<M8[us]',
       'fill_value': 0,
       'filters': (),
       'name': 'time'},
      {'attrs': [('attr', 1)],
       'compressor': None,
       'dimensions': ('num_lines', 'num_pixels'),
       'dtype': '<f8',
       'fill_value': 214748.3647,
       'filters': ({'id': 'fixedscaleoffset',
         'scale': 10000,
         'offset': 0,
         'dtype': '<f8',
         'astype': '<i4'},),
       'name': 'var1'},
      {'attrs': [('attr', 1)],
       'compressor': None,
       'dimensions': ('num_lines', 'num_pixels'),
       'dtype': '<f8',
       'fill_value': 214748.3647,
       'filters': ({'id': 'fixedscaleoffset',
         'scale': 10000,
         'offset': 0,
         'dtype': '<f8',
         'astype': '<i4'},),
       'name': 'var2'})}

Now that the collection has been created, we can insert new records. ::

    >>> collection.insert(ds)

.. note::

    When inserting it’s possible to specify the :ref:`merge strategy of a
    partition <merging_datasets>`. By default, the last inserted data overwrite
    the exising ones. Others strategy can be defined, for example, to update
    existing data (overwrite the updated data, while keeping the existing ones).
    This last strategy allows updating incrementally an existing partition. ::

        >>> import zcollection.merging
        >>> collection.insert(
        ...     ds, merge_callable=zcollection.merging.merge_time_series)

Let's look at the different partitions thus created. ::

    >>> fs.listdir("/my_collection/year=2000")
    [{'name': '/my_collection/year=2000/month=01/',
      'size': 0,
      'type': 'directory'},
     {'name': '/my_collection/year=2000/month=02/',
      'size': 0,
      'type': 'directory'},
     {'name': '/my_collection/year=2000/month=03/',
      'size': 0,
      'type': 'directory'},
     {'name': '/my_collection/year=2000/month=04/',
      'size': 0,
      'type': 'directory'},
     {'name': '/my_collection/year=2000/month=05/',
      'size': 0,
      'type': 'directory'},
     {'name': '/my_collection/year=2000/month=06/',
      'size': 0,
      'type': 'directory'}]

This collection is composed of several partitions, but it is always handled as a
single data set.

Loading data
^^^^^^^^^^^^

To load the dataset call the method
:py:meth:`load<zcollection.collection.Collection.load>` on the instance.  By
default, the method loads all partitions stored in the collection. ::

    >>> collection.load()
    <zcollection.dataset.Dataset>
    Dimensions: "('num_lines: 61', 'num_pixels: 25')"
    Data variables
        time    (num_lines  datetime64[us]: dask.array<chunksize=(11,)>
        var1    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
        var2    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
    Attributes:
        attr   : 1

You can also select the partitions to be considered by filtering the partitions
using keywords used for partitioning. ::

    >>> collection.load("year == 2000 and month == 2")
    <zcollection.dataset.Dataset>
    Dimensions: "('num_lines: 9', 'num_pixels: 25')"
    Data variables
        time    (num_lines  datetime64[us]: dask.array<chunksize=(9,)>
        var1    (num_lines, num_pixels  float64: dask.array<chunksize=(9, 25)>
        var2    (num_lines, num_pixels  float64: dask.array<chunksize=(9, 25)>
    Attributes:
        attr   : 1

Note that the :py:meth:`load<zcollection.collection.Collection.load>` function
may return None if no partition has been selected. ::

    >>> collection.load("year == 2002 and month == 2") is None
    True

Editing variables
^^^^^^^^^^^^^^^^^

.. note::

    The functions for modifying collections are not usable if the collection is
    :py:meth:`open<zcollection.open_collection>` in read-only mode.

It's possible to delete a variable from a collection. ::

    >>> collection.drop_variable("var2")
    >>> collection.load()
    <zcollection.dataset.Dataset>
    Dimensions: "('num_lines: 61', 'num_pixels: 25')"
    Data variables
        time    (num_lines  datetime64[us]: dask.array<chunksize=(11,)>
        var1    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
    Attributes:
        attr   : 1

.. warning::

    The variable used for partitioning cannot be deleted. ::

        >>> collection.drop_variable("time")
        ---------------------------------------------------------------------------
        ValueError                                Traceback (most recent call last)
        <ipython-input-15-a86b16232273> in <module>
        ----> 1 collection.drop_variable("time")

        ...\zcollection\collection.py in drop_variable(self, variable)
            602         _LOGGER.info("Dropping of the %r variable in the collection", variable)
            603         if variable in self.partitioning.variables:
        --> 604             raise ValueError(
            605                 f"The variable '{variable}' is part of the partitioning.")
            606         if variable not in self.metadata.variables:

        ValueError: The variable 'time' is part of the partitioning.

The :py:meth:`add_variable<zcollection.collection.Collection.add_variable>`
method allows you to add a new variable to the collection. ::

    >>> collection.add_variable(ds.metadata().variables["var2"])

The newly created variable is initialized with its default value. ::

    >>> collection.load().variables["var2"].values
    masked_array(
      data=[[--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            ...,
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --]],
      mask=[[ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            ...,
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True]],
      fill_value=214748.3647,
      dtype=float64)

Finally it's possible to
:py:meth:`update<zcollection.collection.Collection.update>` the existing
variables.

In this example, we will alter the variable ``var2`` by setting it to 1 anywhere
the variable ``var1`` is defined. ::

    >>> def ones(ds):
    >>>     return ds.variables["var1"].values * 0 + 1
    >>> collection.update(ones, "var2")
    >>> collection.load().variables["var2"].values
    masked_array(
      data=[[--, --, --, ..., --, --, --],
            [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],
            [--, --, --, ..., --, --, --],
            ...,
            [--, --, --, ..., --, --, --],
            [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],
            [--, --, --, ..., --, --, --]],
      mask=[[ True,  True,  True, ...,  True,  True,  True],
            [False, False, False, ..., False, False, False],
            [ True,  True,  True, ...,  True,  True,  True],
            ...,
            [ True,  True,  True, ...,  True,  True,  True],
            [False, False, False, ..., False, False, False],
            [ True,  True,  True, ...,  True,  True,  True]],
      fill_value=214748.3647)

Views
-----

Creation of views
^^^^^^^^^^^^^^^^^

A :py:class:`view<zcollection.view.View>` allows you to extend a collection
(:py:class:`a view reference<zcollection.view.ViewReference>`) that you are
not allowed to modify. ::

    >>> view = zcollection.create_view("/my_view",
    >>>                                zcollection.view.ViewReference(
    >>>                                    "/my_collection", fs),
    >>>                                filesystem=fs)

.. note::

    The created view can be accessed using the following command ::

        >>> view = zcollection.open_view("/my_view", filesystem=fs)

Editing variables
^^^^^^^^^^^^^^^^^

When the view is created, it has no data of its own, it uses all the data
defined in the reference view. ::

    >>> fs.listdir("/my_view")
    [{'name': '/my_view/.view',
      'size': 414,
      'type': 'file',
      'created': 1634400261.024458}]
    >>> view.load()
    <zcollection.dataset.Dataset>
    Dimensions: "('num_lines: 61', 'num_pixels: 25')"
    Data variables
        time    (num_lines  datetime64[us]: dask.array<chunksize=(11,)>
        var1    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
        var2    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
    Attributes:
        attr   : 1

Such a state of the view is not very interesting. But it is possible to
:py:meth:`add<zcollection.view.View.add_variable>` and modify variables in order
to enhance the view. ::

    >>> var3 = ds.metadata().variables["var2"]
    >>> var3.name = "var3"
    >>> view.add_variable(var3)

This step creates all necessary partitions for the new variable. ::

    >>> fs.listdir("/my_view/year=2000")
    [{'name': '/my_view/year=2000/month=01/', 'size': 0, 'type': 'directory'},
     {'name': '/my_view/year=2000/month=02/', 'size': 0, 'type': 'directory'},
     {'name': '/my_view/year=2000/month=03/', 'size': 0, 'type': 'directory'},
     {'name': '/my_view/year=2000/month=04/', 'size': 0, 'type': 'directory'},
     {'name': '/my_view/year=2000/month=05/', 'size': 0, 'type': 'directory'},
     {'name': '/my_view/year=2000/month=06/', 'size': 0, 'type': 'directory'}]

The new variable is not initialized. ::

    >>> view.load().variables["var3"].values
    masked_array(
      data=[[--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            ...,
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --]],
      mask=[[ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            ...,
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True]],
      fill_value=214748.3647,
      dtype=float64)

The same principle used by the collection allows to
:py:meth:`update<zcollection.view.View.update>` the variables. ::

    >>> view.update(ones, "var3")
    >>> var3 = view.load().variables["var3"].values
    >>> var2 = view.load().variables["var2"].values
    >>> var2 - var3
    masked_array(
      data=[[--, --, --, ..., --, --, --],
            [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
            [--, --, --, ..., --, --, --],
            ...,
            [--, --, --, ..., --, --, --],
            [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
            [--, --, --, ..., --, --, --]],
      mask=[[ True,  True,  True, ...,  True,  True,  True],
            [False, False, False, ..., False, False, False],
            [ True,  True,  True, ...,  True,  True,  True],
            ...,
            [ True,  True,  True, ...,  True,  True,  True],
            [False, False, False, ..., False, False, False],
            [ True,  True,  True, ...,  True,  True,  True]],
      fill_value=214748.3647)

.. warning::

    The variables of the reference collection cannot be edited. ::

        >>> view.update(ones, "var2")
        ---------------------------------------------------------------------------
        ValueError                                Traceback (most recent call last)
        <ipython-input-32-3a170e8da0ec> in <module>
        ----> 1 view.update(ones, "var2")

        ...\zcollection\view.py in update(self, func, variable, filters)
            392         """
            393         _LOGGER.info("Updating variable %r", variable)
        --> 394         _assert_variable_handled(self.view_ref.metadata, self.metadata,
            395                                  variable)
            396         arrays = []

        ...\zcollection\view.py in _assert_variable_handled(reference, view, variable)
            136     """
            137     if variable in reference.variables:
        --> 138         raise ValueError(f"Variable {variable} is read-only")
            139     if variable not in view.variables:
            140         raise ValueError(f"Variable {variable} does not exist")

        ValueError: Variable var2 is read-only


Finally, a method allows you to
:py:meth:`drop_variable<zcollection.view.View.drop_variable>` variables from the
view. ::

    >>> view.load()
    <zcollection.dataset.Dataset>
    Dimensions: "('num_lines: 61', 'num_pixels: 25')"
    Data variables
        time    (num_lines  datetime64[us]: dask.array<chunksize=(11,)>
        var1    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
        var2    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
        var3    (num_lines, num_pixels  float64: dask.array<chunksize=(11, 25)>
    Attributes:
        attr   : 1
    >>> view.drop_variable("var3")

.. warning::

    The variables of the reference collection cannot be deleted. ::

        >>> view.drop_variable("var2")
        ---------------------------------------------------------------------------
        ValueError                                Traceback (most recent call last)
        <ipython-input-33-2a970c7cd699> in <module>
        ----> 1 view.drop_variable("var2")

        .../zcollection/view.py in drop_variable(self, varname)
            310         """
            311         _LOGGER.info("Dropping variable %r", varname)
        --> 312         _assert_variable_handled(self.view_ref.metadata, self.metadata,
            313                                  varname)
            314         client = utilities.get_client()

        .../zcollection/view.py in _assert_variable_handled(reference, view, variable)
            136     """
            137     if variable in reference.variables:
        --> 138         raise ValueError(f"Variable {variable} is read-only")
            139     if variable not in view.variables:
            140         raise ValueError(f"Variable {variable} does not exist")

        ValueError: Variable var2 is read-only

Indexing
--------

A collection can be indexed. This allows quick access to the data without having
to browse the entire dataset.

Creating the test collection.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this latest example, we will index another data set. This one contains
measurements of a fictitious satellite on several half-orbits. ::

    >>> ds = zcollection.Dataset.from_xarray(
    ...         zcollection.tests.data.create_test_sequence(5, 20, 10))
    <zcollection.dataset.Dataset>
      Dimensions: "('num_lines: 1000',)"
    Data variables
        time         (num_lines  datetime64[ns]: dask.array<chunksize=(1000,)>
        cycle_number (num_lines  int64: dask.array<chunksize=(1000,)>
        pass_number  (num_lines  int64: dask.array<chunksize=(1000,)>
        observation  (num_lines  float64: dask.array<chunksize=(1000,)>

::

    >>> collection = zcollection.create_collection(
    ...         "time",
    ...         ds,
    ...         zcollection.partitioning.Date(("time", ), "M"),
    ...         partition_base_dir=str("/one_other_collection"),
    ...         filesystem=fs)
    ... collection.insert(ds, zcollection.merging.merge_time_series)

Here we have created a collection partitioned by month. ::

    >>> fs.listdir("/one_other_collection/year=2000")
    [{'name': '/one_other_collection/year=2000/month=01',
      'size': 0,
      'type': 'directory'},
     {'name': '/one_other_collection/year=2000/month=02',
      'size': 0,
      'type': 'directory'},
     {'name': '/one_other_collection/year=2000/month=03',
      'size': 0,
      'type': 'directory'}]

Class to implement
^^^^^^^^^^^^^^^^^^

The idea of the implementation is to calculate for each visited partition, the
slice of data that has a constant quantity. In our example, we will rely on the
cycle and pass number information. The first method we will implement is the
detection of these constant parts of two vectors containing the cycle and pass
number.

.. code-block:: python

    from typing import Iterator, Tuple

    import numpy


    def split_half_orbit(
        cycle_number: numpy.ndarray,
        pass_number: numpy.ndarray,
    ) -> Iterator[Tuple[int, int]]:
        """
        Calculate the indexes of the start and stop of each half-orbit.
        Args:
            pass_number: Pass numbers.
        Returns:
            Iterator of start and stop indexes.
        """
        assert pass_number.shape == cycle_number.shape
        pass_idx = numpy.where(numpy.roll(pass_number, 1) != pass_number)[0]
        cycle_idx = numpy.where(numpy.roll(cycle_number, 1) != cycle_number)[0]

        half_orbit = numpy.unique(
            numpy.concatenate(
                (pass_idx, cycle_idx, numpy.array([pass_number.size],
                                                  dtype="int64"))))
        del pass_idx, cycle_idx

        for idx0, idx1 in tuple(zip(half_orbit[:-1], half_orbit[1:])):
            yield idx0, idx1

Now we will compute these constant parts from a dataset contained in a
partition.

.. code-block:: python

    def _half_orbit(
        ds: zcollection.Dataset,
        *args,
        **kwargs,
    ) -> numpy.ndarray:
        """
        Return the indexes of the start and stop of each half-orbit.
        Args:
            ds: Datasets stored in a partition to be indexed.
        Returns:
            Dictionary of start and stop indexes for each half-orbit.
        """
        pass_number_varname = kwargs.pop('pass_number', 'pass_number')
        cycle_number_varname = kwargs.pop('cycle_number', 'cycle_number')
        pass_number = ds.variables[pass_number_varname].values
        cycle_number = ds.variables[cycle_number_varname].values

        generator = ((
            i0,
            i1,
            cycle_number[i0],
            pass_number[i0],
        ) for i0, i1 in split_half_orbit(cycle_number, pass_number))

        return numpy.fromiter(
            generator, numpy.dtype(HalfOrbitIndexer.dtype()))

Finally, we implement our indexing class. The base class
(:py:class:`zcollection.indexing.Indexer<zcollection.indexing.abc.Indexer>`)
implements the index update and the associated queries.

.. code-block:: python

    import pathlib
    from typing import List, Optional, Union

    import zcollection.indexing


    class HalfOrbitIndexer(zcollection.indexing.Indexer):
        """Index collection by half-orbit.
        """
        #: Column name of the cycle number.
        CYCLE_NUMBER = "cycle_number"

        #: Column name of the pass number.
        PASS_NUMBER = "pass_number"

        @classmethod
        def dtype(cls, /, **kwargs) -> List[Tuple[str, str]]:
            """Return the columns of the index.
            Returns:
                A tuple of (name, type) pairs.
            """
            return super().dtype() + [
                (cls.CYCLE_NUMBER, "uint16"),
                (cls.PASS_NUMBER, "uint16"),
            ]

        @classmethod
        def create(
            cls,
            path: Union[pathlib.Path, str],
            ds: zcollection.Collection,
            filesystem: Optional[fsspec.AbstractFileSystem] = None,
            **kwargs,
        ) -> "HalfOrbitIndexer":
            """Create a new index.
            Args:
                path: The path to the index.
                ds: The collection to be indexed.
                filesystem: The filesystem to use.
            Returns:
                The created index.
            """
            return super()._create(path,
                                   ds,
                                   meta=dict(attribute=b"value"),
                                   filesystem=filesystem)  # type: ignore

        def update(
            self,
            ds: zcollection.Collection,
            bag_partition_size: Optional[int] = None,
            bag_npartitions: Optional[int] = None,
            **kwargs,
        ) -> None:
            """
            Update the index.
            Args:
                ds: New data stored in the collection to be indexed.
                bag_partition_size: The length of each bag partition.
                bag_npartitions: The number of desired bag partitions.
                cycle_number: The name of the cycle number variable stored in the
                    collection. Defaults to "cycle_number".
                pass_number: The name of the pass number variable stored in the
                    collection. Defaults to "pass_number".
            """
            super()._update(ds, _half_orbit, bag_partition_size, bag_npartitions,
                            **kwargs)

Using the index
^^^^^^^^^^^^^^^

Now we can create our index and fill it. ::

    >>> indexer = HalfOrbitIndexer.create("/index.parquet", collection, filesystem=fs)
    >>> indexer.update(collection)

The following command allows us to view the information stored in our index: the
first and last indexes of the partition associated with the registered
half-orbit number and the identifier of the indexed partition. ::

    >>> indexer.table.to_pandas()
        start  stop  cycle_number  pass_number  year  month
    0       0    20             1            1  2000      1
    1      20    40             1            2  2000      1
    2      40    60             1            3  2000      1
    3      60    80             1            4  2000      1
    4      80   100             1            5  2000      1
    5     100   120             2            1  2000      1
    6     120   140             2            2  2000      1
    7     140   160             2            3  2000      1
    8     160   180             2            4  2000      1
    9     180   200             2            5  2000      1
    10    200   220             3            1  2000      1
    11    220   240             3            2  2000      1
    12    240   260             3            3  2000      1
    13    260   280             3            4  2000      1
    14    280   300             3            5  2000      1
    15    300   320             4            1  2000      1
    16    320   340             4            2  2000      1
    17    340   360             4            3  2000      1
    18    360   372             4            4  2000      1
    19      0     8             4            4  2000      2
    20      8    28             4            5  2000      2
    21     28    48             5            1  2000      2
    22     48    68             5            2  2000      2
    23     68    88             5            3  2000      2
    …
    46    180   200            10            1  2000      3
    47    200   220            10            2  2000      3
    48    220   240            10            3  2000      3
    49    240   260            10            4  2000      3
    50    260   280            10            5  2000      3

This index can now be used to load a part of a collection. ::

    >>> selection = collection.load(indexers=indexer.query(dict(pass_number=[1, 2])))
    >>> selection.to_xarray().compute()
    <xarray.Dataset>
    Dimensions:       (num_lines: 400)
    Dimensions without coordinates: num_lines
    Data variables:
        cycle_number  (num_lines) float64 1.0 1.0 1.0 1.0 ... 10.0 10.0 10.0 10.0
        observation   (num_lines) float64 0.943 0.02556 0.6605 ... 0.8343 0.5307
        pass_number   (num_lines) float64 1.0 1.0 1.0 1.0 1.0 ... 2.0 2.0 2.0 2.0
        time          (num_lines) datetime64[ns] 2000-01-01 ... 2000-03-19T06:00:00
