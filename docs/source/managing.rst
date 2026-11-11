Managing variables and partitions
=================================

:py:meth:`~zcollection.collection.Collection.drop_partitions` removes
partitions from a collection.

The examples use daily measurements split into segments of three, two and
four samples.

.. jupyter-execute::

    import pathlib
    import tempfile

    import numpy
    import zcollection

    root = pathlib.Path(tempfile.mkdtemp())
    origin = numpy.datetime64('2026-01-01', 'D')
    data = zcollection.Dataset([
        zcollection.Array(
            'time', origin + numpy.arange(9).astype('timedelta64[D]'),
            ('sample', )),
        zcollection.Array(
            'segment', numpy.repeat([1, 2, 3], [3, 2, 4]), ('sample', )),
        zcollection.Array(
            'signal', numpy.arange(9, dtype='float64') ** 2, ('sample', )),
    ], delayed=False)
    collection = zcollection.create_collection(
        'time', data, zcollection.partitioning.Sequence(('segment', )),
        str(root / 'collection'))
    collection.insert(data, distributed=False)
    list(collection.partitions(relative=True))

Dropping partitions
-------------------

Partitions can be selected using ``filters``, ``timedelta``, or both:

- ``filters`` selects partitions based on their partition keys.
- ``timedelta`` selects partitions older than the specified duration, based
  on timestamps reported by the file system.

When both parameters are provided, only partitions matching both conditions
are removed. If neither is specified, all partitions are removed, while the
configuration of the collection and its immutable variables are preserved.

For example, to remove the partition where ``segment == 2``:

.. jupyter-execute::

    collection.drop_partitions(filters='segment == 2', distributed=False)

.. jupyter-execute::

    list(collection.partitions(relative=True))

The method returns the paths of the removed partitions, relative to the
collection. Partitions listed in the index but missing from the store are
excluded from the returned paths.

Timestamps of the file system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **S3:** creation timestamps are not available, so using ``timedelta``
  raises ``NotImplementedError``.
- **Local file systems on Linux:** the reported timestamp reflects the last
  change to the partition directory rather than its actual creation time.
  Inserting data into a partition, for example, resets its age.

Distributed execution
^^^^^^^^^^^^^^^^^^^^^

By default, ``distributed=True`` removes partitions using separate Dask
tasks, with up to ``npartitions`` tasks running concurrently.

See :doc:`failures` for details on how errors are handled during partition
removal.
