Getting started
===============

On this page you declare a collection, insert two batches of measurements into
it, read them back and add a variable.

The collection holds a swath: a line of measurements per date, a longitude
across it, and the sea surface height at each point.

The schema
----------

You declare a collection before it holds anything. The schema describes the
variables, their data types, the dimensions they span, and the way each one
is stored.

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

The dimensions and the encoding of each variable apply for the life of the
collection.

Dimensions
    ``num_lines`` is the dimension the data is partitioned along: it grows
    as measurements arrive, and the collection records its size as ``-1``. Every
    other dimension has the size declared here, and a dataset inserted with
    another size is refused.

    ``lon`` spans ``num_pixels`` alone, so its values are the same in every
    partition and the collection stores them once. :doc:`index` shows the
    layout that follows.

``fill_value``
    The value that stands for an absent measurement, written in the unit of the
    variable. A read masks the elements equal to it.

``filters`` and ``compressor``
    The encoding that turns the values into bytes.
    :py:class:`FixedScaleOffset <numcodecs.fixedscaleoffset.FixedScaleOffset>`
    encodes ``ssh`` to the millimetre as a 16-bit integer, and
    :py:class:`Blosc <numcodecs.blosc.Blosc>` compresses the result. The
    collection applies both to every partition.

``attrs``
    The information the variable carries, its unit here. The metadata holds
    it, and a read returns it.

Creating the collection
-----------------------

A collection needs four things: its schema, the name of the partitioning
variable, the :py:mod:`partitioning <zcollection.partitioning>` itself, and
the directory that holds the partitions.

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

No partition exists yet. The schema is now in the ``.zcollection`` file of
that directory, and
:py:func:`open_collection <zcollection.open_collection>` reads it back.

Inserting data
--------------

The data carries the values, the names and the dimensions; the collection
supplies the fill value, the encoding and the attributes.

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

    collection.insert(swath('2022-01-01', 59), distributed=False)
    list(collection.partitions(relative=True))

:py:meth:`insert <zcollection.collection.Collection.insert>` splits the
dataset on the partitioning variable and writes one partition per month. Pass
``distributed=False`` to keep the work in the calling process, as every
example here does.

If partition listings are inconsistent, see :doc:`partition_index` for
instructions on rebuilding the index.

A second insertion adds the partition its dates fall in.

.. jupyter-execute::

    collection.insert(swath('2022-04-01', 30), distributed=False)
    list(collection.partitions(relative=True))

Reading
-------

:py:meth:`load <zcollection.collection.Collection.load>` concatenates the
partitions along ``num_lines``, the dimension of the partitioning variable,
and returns one dataset.

.. jupyter-execute::

    loaded = collection.load(distributed=False)
    print(loaded.variables['time'].values[[0, -1]])
    loaded

The result holds ``lon`` with its 5 values, and ``ssh`` carries the unit
declared for it.

.. jupyter-execute::

    print(loaded.variables['lon'].values)
    [(item.name, item.value) for item in loaded.variables['ssh'].attrs]

The filter declared on ``ssh`` encodes it to the millimetre, so a read gives
back a rounded height.

.. jupyter-execute::

    print(swath('2022-01-01', 59).variables['ssh'].values[0, :3])
    loaded.variables['ssh'].values[0, :3]

:py:meth:`load <zcollection.collection.Collection.load>` reads the
partitions whose keys satisfy its ``filters`` argument. The keys are those of
the partitioning: ``year`` and ``month`` here.

.. jupyter-execute::

    loaded = collection.load(filters='year == 2022 and month == 4',
                             distributed=False)
    loaded.variables['ssh'].shape

Its ``selected_partitions`` argument takes the path of each partition
instead, relative to ``base_dir``.

.. jupyter-execute::

    loaded = collection.load(selected_partitions=['year=2022/month=01'],
                             distributed=False)
    loaded.variables['ssh'].shape

A selection that matches no partition loads nothing.

.. jupyter-execute::

    print(collection.load(filters='year == 1999', distributed=False))

Adding and updating a variable
------------------------------

:py:meth:`add_variable <zcollection.collection.Collection.add_variable>`
adds the variable to every partition and fills it with its fill value.

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

:py:meth:`update <zcollection.collection.Collection.update>` computes those
values. The callable receives one partition and returns the values for it, so
the mean below is the mean of one month.

.. jupyter-execute::

    def anomaly(zds: zcollection.Dataset) -> dict[str, numpy.ndarray]:
        """Compute the departure of a height from the mean of its partition."""
        height = zds.variables['ssh'].values
        return {'anomaly': (height - height.mean()).astype('float32')}

    collection.update(anomaly, distributed=False)

    loaded = collection.load(selected_partitions=['year=2022/month=01'],
                             distributed=False)
    loaded.variables['anomaly'].values[0, :3]

Where to go next
----------------

:doc:`partitioning`
    Date against sequence, the resolution and what it costs.

:doc:`auto_examples/ex_collection`
    Dropping partitions and variables, mapping a function over a collection.

:doc:`auto_examples/ex_view`
    Holding variables of your own beside a collection you read.

:doc:`auto_examples/ex_indexing`
    Building an index to read a part of a collection without opening it.

:doc:`auto_examples/ex_without_dask`
    Reading and writing with no Dask cluster, and computing a bag.
