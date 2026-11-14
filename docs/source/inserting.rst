Inserting data
==============

:py:meth:`~zcollection.collection.Collection.insert` splits a dataset
according to the collection's partitioning scheme and writes the data
to the corresponding partitions.

The following examples use a collection partitioned by ``segment``.
Unless otherwise specified, operations run in the calling process with
``distributed=False``.

.. jupyter-execute::

    import pathlib
    import tempfile

    import numpy
    import zcollection

    root = pathlib.Path(tempfile.mkdtemp())
    origin = numpy.datetime64('2026-01-01', 'h')

    def measures(first_hour, count, segment, value):
        """Return hourly measurements of one segment, all equal to value."""
        hours = first_hour + numpy.arange(count)
        return zcollection.Dataset([
            zcollection.Array('time', origin + hours.astype('timedelta64[h]'),
                              ('sample', )),
            zcollection.Array('segment', numpy.full(count, segment),
                              ('sample', )),
            zcollection.Array('signal', numpy.full(count, float(value)),
                              ('sample', )),
        ], delayed=False)

    collection = zcollection.create_collection(
        'time', measures(0, 1, 1, 0),
        zcollection.partitioning.Sequence(('segment', )),
        str(root / 'collection'))


Inserting a dataset
-------------------

First, insert a dataset containing four measurements for segment 1 and
two for segment 2.

.. jupyter-execute::

    first = measures(0, 4, 1, 1).concat(measures(4, 2, 2, 1), 'sample')
    collection.insert(first, distributed=False)

The insertion creates one partition per segment and returns their paths
relative to the collection.

You can retrieve these paths using
:py:meth:`~zcollection.collection.Collection.partitions`:

.. jupyter-execute::

    list(collection.partitions(relative=True))

Partition paths can also be passed to ``selected_partitions`` in
operations such as ``load`` and ``update``.

If one or more partitions fail during insertion, ``insert`` raises
:py:class:`~zcollection.PartitionErrors`. See :doc:`failures` for details.


Inserting into existing partitions
----------------------------------

By default, inserting data into an existing partition replaces its
entire contents.

For example, insert four measurements for segment 1, covering hours
2 to 5, all with the value 2:

.. jupyter-execute::

    second = measures(2, 4, 1, 2)
    collection.insert(second, distributed=False)

    def show(segment):
        """Return the hours and values stored in a segment."""
        loaded = collection.load(selected_partitions=[f'segment={segment}'],
                                 delayed=False, distributed=False)
        hours = (loaded['time'].values - origin).astype(int)
        return list(zip(hours.tolist(), loaded['signal'].values.tolist()))

    show(1)

Segment 1 now contains only the four new measurements. The original
measurements at hours 0 and 1 have also been removed, even though they
do not overlap with the inserted data.


Merging with existing data
~~~~~~~~~~~~~~~~~~~~~~~~~~

To preserve existing data, provide a merge function through the
``merge_callable`` parameter.

ZCollection provides
:py:func:`~zcollection.merging.time_series.merge_time_series`, which
replaces existing measurements within the time span of the inserted
data while keeping those outside it.

To illustrate this behavior, insert the first dataset again, then
insert the second one using the merge function:

.. jupyter-execute::

    collection.insert(first, distributed=False)
    collection.insert(second,
                      merge_callable=zcollection.merging.merge_time_series,
                      distributed=False)
    show(1)

Segment 1 now retains hours 0 and 1 from the original data, while
hours 2 to 5 contain the new measurements.

Additional keyword arguments passed to ``insert`` are forwarded to the
merge function, provided they are not parameters of ``insert`` itself.
For example, ``merge_time_series`` accepts a ``tolerance`` argument.

Be careful with arguments already supplied positionally to the merge
function. Passing one of them as a keyword, such as ``axis``, causes
each affected partition to fail with a ``TypeError``, reported through
:py:class:`~zcollection.PartitionErrors`.


Inserting with Dask
-------------------

By default, ``insert`` uses a Dask cluster (``distributed=True``).
Each partition is written in a separate task on the cluster associated
with the current Dask client.

The ``npartitions`` parameter controls how many partition tasks can run
concurrently. Initially, ``insert`` submits up to ``npartitions`` tasks,
then submits another whenever one finishes.

The default is ``npartitions=1``. Setting it to ``None`` uses the number
of Dask workers instead.

For example, to insert data using a cluster with four workers and up
to eight concurrent partition tasks:

.. code-block:: python

    from dask.distributed import Client

    with Client(n_workers=4):
        collection.insert(first, npartitions=8)

Within each task, the variables of a partition are written in parallel
using the threads of its Dask worker.
