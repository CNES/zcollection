Getting started
===============

This guide walks you through the basics of ZCollection: defining a collection,
inserting data, reading it back, and adding a new variable.

You'll use a swath dataset containing one line of measurements per day.
Each line has five pixels, with a longitude and a sea surface height
measurement at each pixel.


Defining the schema
-------------------

Before creating a collection, you need to define its schema. This describes
the variables, their data types, dimensions, and how their values are stored.

.. jupyter-execute::

    import numcodecs
    import numpy

    import zcollection
    from zcollection import meta

    schema = meta.DatasetSchema(
        dimensions=(meta.Dimension('num_lines', -1),
                    meta.Dimension('num_pixels', 5)),
        variables=(
            meta.VariableSchema(name='time',
                          dtype='datetime64[s]',
                          dimensions=('num_lines', )),
            meta.VariableSchema(name='lon',
                          dtype='float32',
                          dimensions=('num_pixels', ),
                          attrs=(meta.Attribute('units', 'degrees_east'), )),
            meta.VariableSchema(name='ssh',
                          dtype='float32',
                          dimensions=('num_lines', 'num_pixels'),
                          fill_value=-32.0,
                          filters=(numcodecs.FixedScaleOffset(offset=0,
                                                              scale=1000,
                                                              dtype='f4',
                                                              astype='i2'), ),
                          compressor=numcodecs.Blosc(cname='zstd', clevel=5),
                          attrs=(meta.Attribute('units', 'm'), )),
        ))

    sorted(schema.variables)

The dimensions and encoding of each variable remain fixed throughout the
collection's lifetime:

Dimensions
    ``num_lines`` is the dimension along which the collection is partitioned.
    Its size is set to ``-1`` because it grows as new measurements are added.
    All other dimensions have fixed sizes, and inserted datasets must
    respect them.

    The ``lon`` variable depends only on ``num_pixels``. Its values are
    therefore identical across partitions and are stored only once.
    See :doc:`index` for an example of the resulting storage layout.

``fill_value``
    Represents missing measurements, expressed in the variable's units.
    Values equal to ``fill_value`` are masked when the data is read.

``filters`` and ``compressor``
    Control how values are encoded and compressed before storage.

    :py:class:`FixedScaleOffset <numcodecs.fixedscaleoffset.FixedScaleOffset>`
    encodes ``ssh`` values with millimetre precision as 16-bit integers.
    :py:class:`Blosc <numcodecs.blosc.Blosc>` then compresses the encoded
    data. Both are applied to every partition.

``attrs``
    Stores additional information about a variable, such as its units.
    These attributes are preserved in the collection's metadata and
    restored when the data is read.


Creating the collection
-----------------------

Creating a collection requires a schema, a partitioning variable, a
:py:mod:`partitioning scheme <zcollection.partitioning>`, and a directory
where the data will be stored.

Here, the collection is partitioned by month using the ``time`` variable.

.. jupyter-execute::

    import pathlib
    import tempfile

    base_dir = str(pathlib.Path(tempfile.mkdtemp()) / 'my_collection')

    collection = zcollection.Collection(
        'time',
        schema,
        zcollection.partitioning.Date(('time', ), resolution='M'),
        base_dir,
        mode='w')

    list(collection.partitions(relative=True))

The collection is now initialized, but contains no partitions. Its schema
is stored in the ``.zcollection`` file and can be read later using
:py:func:`open_collection <zcollection.open_collection>`.


Inserting data
--------------

To insert data, provide a dataset containing the variable names, dimensions,
and values. The collection uses its schema to determine the fill values,
encoding, and attributes.

First, create a function that generates a swath with five measurements
per day.

.. jupyter-execute::

    def swath(start: str, days: int) -> zcollection.Dataset:
        """Build one line of five measurements per day."""
        time = numpy.arange(
            numpy.datetime64(start),
            numpy.datetime64(start) + numpy.timedelta64(days, 'D'),
            numpy.timedelta64(1, 'D')).astype('datetime64[s]')
        height = numpy.linspace(-1.5, 1.5, days * 5).reshape(days, 5)
        return zcollection.Dataset(variables=[
            zcollection.Array('time', time, ('num_lines', )),
            zcollection.Array('lon',
                              numpy.linspace(-10, 10, 5).astype('float32'),
                              ('num_pixels', )),
            zcollection.Array('ssh', height.astype('float32'),
                              ('num_lines', 'num_pixels')),
        ])

Insert 59 days of measurements, covering January and February 2022:

.. jupyter-execute::

    collection.insert(swath('2022-01-01', 59), distributed=False)
    list(collection.partitions(relative=True))

:py:meth:`insert <zcollection.collection.Collection.insert>` uses the
partitioning scheme to split the dataset into monthly partitions.

``distributed=False`` runs the operation without a Dask cluster, as in
every example in this guide.

Now insert another batch of measurements for April:

.. jupyter-execute::

    collection.insert(swath('2022-04-01', 30), distributed=False)
    list(collection.partitions(relative=True))

The collection now contains three monthly partitions.


Reading data
------------

:py:meth:`load <zcollection.collection.Collection.load>` reads the
partitions and combines them into a single dataset, concatenating their
data along ``num_lines``.

.. jupyter-execute::

    loaded = collection.load(distributed=False)
    print(loaded.variables['time'].values[[0, -1]])
    loaded

The resulting dataset includes the five longitude values stored in
``lon``, along with the attributes defined for ``ssh``.

.. jupyter-execute::

    print(loaded.variables['lon'].values)
    [(item.name, item.value) for item in loaded.variables['ssh'].attrs]

The encoding defined in the schema is also applied transparently.
Because ``ssh`` is stored with millimetre precision, the values read
back may differ slightly from the originals.

.. jupyter-execute::

    print(swath('2022-01-01', 59).variables['ssh'].values[0, :3])
    loaded.variables['ssh'].values[0, :3]

Selecting partitions
^^^^^^^^^^^^^^^^^^^^

You can load only part of a collection by selecting its partitions.

The ``filters`` argument selects partitions using their keys. With the
monthly partitioning used here, these are ``year`` and ``month``.

For example, to load only April 2022:

.. jupyter-execute::

    loaded = collection.load(filters='year == 2022 and month == 4',
                             distributed=False)
    loaded.variables['ssh'].shape

Alternatively, ``selected_partitions`` accepts a list of partition paths
relative to ``base_dir``:

.. jupyter-execute::

    loaded = collection.load(selected_partitions=['year=2022/month=01'],
                             distributed=False)
    loaded.variables['ssh'].shape

If no partition matches the selection, ``load`` returns ``None``:

.. jupyter-execute::

    print(collection.load(filters='year == 1999', distributed=False))


Adding and updating a variable
------------------------------

You can add variables to an existing collection without recreating it.

:py:meth:`add_variable <zcollection.collection.Collection.add_variable>`
adds a variable to every partition, initially filling it with the
specified fill value.

Here, add an ``anomaly`` variable with the same dimensions as ``ssh``:

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema(name='anomaly',
                      dtype='float32',
                      dimensions=('num_lines', 'num_pixels'),
                      fill_value=-32.0,
                      attrs=(meta.Attribute('units', 'm'), )),
        distributed=False)

    loaded = collection.load(selected_partitions=['year=2022/month=01'],
                             distributed=False)
    loaded.variables['anomaly'].values[0, :3]

To populate the variable, use
:py:meth:`update <zcollection.collection.Collection.update>` with a
function that computes its values.

The function is called separately for each partition. In this example,
the difference between each sea surface height measurement and the mean
of its partition is calculated.

.. jupyter-execute::

    def anomaly(zds: zcollection.Dataset) -> dict[str, numpy.ndarray]:
        """Compute the departure of a height from the mean of its partition."""
        height = zds.variables['ssh'].values
        return {'anomaly': (height - height.mean()).astype('float32')}

    collection.update(anomaly, distributed=False)

    loaded = collection.load(selected_partitions=['year=2022/month=01'],
                             distributed=False)
    loaded.variables['anomaly'].values[0, :3]

Because each partition represents one month, the mean is calculated
independently for each month.


Where to go next
----------------

:doc:`partitioning`
    Explore date and sequence partitioning, resolution choices, and their
    implications.

:doc:`inserting`
    Learn how to replace or merge data in existing partitions and perform
    insertions using Dask.

:doc:`computing`
    Apply functions to partitions, with or without data from neighboring
    partitions.

:doc:`updating`
    Update variables using computed values or data from another dataset.

:doc:`managing`
    Add and remove variables, and remove partitions.

:doc:`failures`
    Understand how partition failures are reported and how to retry
    failed operations.

:doc:`auto_examples/ex_collection`
    Follow a collection from creation to mapping: loading, filtering,
    adding, updating and removing variables.

:doc:`auto_examples/ex_view`
    Create views to store variables derived from an existing collection.

:doc:`auto_examples/ex_indexing`
    Build an index to access part of a collection without opening it.

:doc:`auto_examples/ex_without_dask`
    Read and write collections without a Dask cluster.
