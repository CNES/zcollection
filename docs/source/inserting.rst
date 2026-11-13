Inserting data
==============

:py:meth:`~zcollection.collection.Collection.insert` splits a dataset on the
partitioning variables and writes each slice to its partition. The examples
use a collection partitioned by ``segment`` and run each operation in the
calling process.

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

Insert a dataset holding four measurements of segment 1 and two of segment 2.
``insert`` writes one partition per segment and returns their paths,
relative to the collection.

.. jupyter-execute::

    first = measures(0, 4, 1, 1).concat(measures(4, 2, 2, 1), 'sample')
    collection.insert(first, distributed=False)

:py:meth:`~zcollection.collection.Collection.partitions` lists the same
paths, and ``selected_partitions`` takes them, for example in ``load`` and
``update``.

.. jupyter-execute::

    list(collection.partitions(relative=True))

If a partition fails, ``insert`` raises
:py:class:`~zcollection.PartitionErrors`; see :doc:`failures`.

Inserting into existing partitions
----------------------------------

By default, the data inserted into a partition replaces the data it holds.
Insert a dataset covering hours 2 to 5 of segment 1, with the value 2: segment
1 then holds these four measurements only.

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

A merge callable combines the stored data with the inserted data instead.
:py:func:`~zcollection.merging.time_series.merge_time_series` replaces the
stored measurements that fall within the time span of the inserted ones and
keeps the others. The cell below inserts the first dataset again, then the
second one with this callable; segment 1 keeps hours 0 and 1 from the
first:

.. jupyter-execute::

    collection.insert(first, distributed=False)
    collection.insert(second,
                      merge_callable=zcollection.merging.merge_time_series,
                      distributed=False)
    show(1)

The keyword arguments of ``insert`` that are not its own parameters reach the
merge callable, such as the ``tolerance`` of ``merge_time_series``. An
argument named after a positional parameter of the merge callable, such as
``axis``, makes each partition that already holds data fail with
``TypeError``.

Inserting with Dask
-------------------

With ``distributed=True``, the default, ``insert`` runs on the Dask cluster
of the current client. Each partition is written in a task of its own:
``insert`` submits ``npartitions`` tasks, then one more each time a task
finishes. ``npartitions`` defaults to 1; with ``None``, it is the number of
Dask workers.

.. code-block:: python

    from dask.distributed import Client

    with Client(n_workers=4):
        collection.insert(first, npartitions=8)

A task writes the variables of its partition in parallel, using the threads
of its worker.
