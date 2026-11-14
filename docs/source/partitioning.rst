Choosing a partitioning
=======================

A :py:mod:`partitioning scheme <zcollection.partitioning>` determines how
a dataset is divided into partitions. It is defined when the collection is
created and stored in its configuration, so the same scheme is used
throughout the collection's lifetime.

Each partition is identified by a key made up of ``name=value`` pairs,
such as ``year=2022/month=01``. These keys determine both how partitions
are organized and how they can be selected.


By date
-------

:py:class:`Date <zcollection.partitioning.date.Date>` partitions a dataset
based on a date variable. It supports six resolutions, from years down to
seconds, each determining the components of the partition key.

The following example shows the keys generated for the same date at each
resolution:

.. jupyter-execute::

    import numpy

    import zcollection
    from zcollection import partitioning

    instant = numpy.array(['2022-06-15T10:30:05'], dtype='datetime64[s]')
    zds = zcollection.Dataset(
        variables=[zcollection.Array('time', instant, ('time', ))])

    for resolution in ('Y', 'M', 'D', 'h', 'm', 's'):
        scheme = partitioning.Date(('time', ), resolution=resolution)
        keys, _ = next(iter(scheme.split_dataset(zds, 'time')))
        print(resolution, '->', '/'.join(keys))

Date partitioning requires exactly one date variable and one of the six
supported resolutions. Multiple variables, unsupported resolutions, or
incompatible variable types raise errors.


By sequence
-----------

:py:class:`Sequence <zcollection.partitioning.sequence.Sequence>` groups
measurements according to one or more variables whose values together
form a monotonic key.

For example, satellite measurements can be partitioned by cycle and pass
number.


One partition per key
~~~~~~~~~~~~~~~~~~~~~

With ``Sequence``, each distinct combination of values creates a separate
partition.

.. jupyter-execute::

    passes = zcollection.Dataset(variables=[
        zcollection.Array('cycle', numpy.repeat([1, 2], 6), ('n', )),
        zcollection.Array('pass', numpy.tile([1, 2, 3, 4, 5, 6], 2), ('n', )),
    ])

    scheme = partitioning.Sequence(('cycle', 'pass'))
    keys = ['/'.join(item) for item, _ in scheme.split_dataset(passes, 'n')]
    len(keys), keys[0], keys[-1]

This approach creates one partition for every distinct key. A collection
containing thousands of satellite passes can therefore have thousands
of partitions.


Grouping keys into partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`GroupedSequence
<zcollection.partitioning.sequence.GroupedSequence>` reduces the number
of partitions by grouping values of the last variable into bins. Each
partition then covers a range of values rather than a single value.

Two parameters control the grouping:

- ``size`` defines the number of values covered by each bin.
- ``start`` determines where the bins are aligned.

The last component of each partition key contains the lower bound of
its bin.

The following example shows how changing ``start`` affects the resulting
partitions:

.. jupyter-execute::

    for start in (1, 0):
        scheme = partitioning.GroupedSequence(('cycle', 'pass'),
                                              size=3,
                                              start=start)
        keys = [item for item, _ in scheme.split_dataset(passes, 'n')]
        bins = sorted({item[1] for item in keys})
        print(f'start={start}: {len(keys)} partitions, bins {bins}')

The ``dtype`` parameter specifies the data types used for partition keys,
which may differ from those of the input variables.

Each bin's lower bound must fit within its declared key type. Otherwise,
splitting raises ``ValueError``. If the alignment can produce negative
bounds, use a signed key type, even when the input values are unsigned.


Selecting partitions by key
---------------------------

Partition keys are also used to select which partitions an operation
should read or update.

The ``filters`` argument accepts expressions based on the names defined
by the partitioning scheme:

.. code-block:: python

    collection.load(filters='year == 2022 and month == 6')
    collection.load(filters='cycle == 2 and pass > 40')

Only the keys defined by the partitioning scheme can be used in these
expressions. For example, a collection partitioned by year has no
``month`` key, so individual months cannot be selected.

The partitioning scheme therefore determines not only how data is
organized, but also how precisely operations can target it.


Choosing a resolution
---------------------

Partitioning involves a trade-off between the amount of data read by
an operation and the number of partitions maintained.

Finer resolutions allow operations to target smaller amounts of data,
but create more partitions and increase metadata overhead.

Consider one year of hourly measurements (8,760 points), with a
``datetime64[ns]`` time variable and a ``float32`` measurement variable.
Together, they account for 105,120 bytes of array data, stored at three
different resolutions:

=============  ==========  ====================  =========================
Resolution     Partitions  Metadata, in bytes    Points read for one day
=============  ==========  ====================  =========================
``Y``                   1                 4 364                     8 760
``M``                  12                38 295                       720
``D``                 365             1 124 030                        24
=============  ==========  ====================  =========================

The amount of data remains the same at every resolution, but the
metadata grows significantly as the number of partitions increases.

With daily partitioning, the 105,120 bytes of data account for only
about 9% of the collection's total size. Most of the remaining space
is occupied by the ``zarr.json`` metadata files associated with each
partition and its variables.

Choose a resolution that matches the amount of data your operations
typically need to process, while keeping the number of partitions
manageable.

The same trade-off applies to sequence partitioning. ``Sequence`` creates
one partition per distinct key, while ``GroupedSequence`` reduces the
number of partitions by grouping values of the last variable without
changing the variables used for partitioning.
