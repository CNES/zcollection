Choosing a partitioning
=======================

The :py:mod:`partitioning <zcollection.partitioning>` splits a dataset into
the partitions a collection is made of. You choose it when you create the
collection, and :py:func:`open_collection <zcollection.open_collection>`
reads it from the configuration, so the choice stands for the life of the
collection.

A partitioning gives each partition a key: the ``name=value`` pairs its
directory carries, such as ``year=2022/month=01``.

By date
-------

:py:class:`Date <zcollection.partitioning.date.Date>` reads one variable of
dates and splits it at one of six resolutions. The resolution sets how many
pairs the key carries.

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

Several variables, a resolution outside the six, or a variable of another
type each stop with an error.

By sequence
-----------

:py:class:`Sequence <zcollection.partitioning.sequence.Sequence>` reads
several variables whose values together form a monotonic key: the cycle and
the pass number of a satellite, for one.

One partition per key
~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    passes = zcollection.Dataset(variables=[
        zcollection.Array('cycle', numpy.repeat([1, 2], 6), ('n', )),
        zcollection.Array('pass', numpy.tile([1, 2, 3, 4, 5, 6], 2), ('n', )),
    ])

    scheme = partitioning.Sequence(('cycle', 'pass'))
    keys = ['/'.join(item) for item, _ in scheme.split_dataset(passes, 'n')]
    len(keys), keys[0], keys[-1]

Every distinct key makes a partition, so a collection holding thousands of
passes is split into thousands of partitions.

One partition per group of keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`GroupedSequence
<zcollection.partitioning.sequence.GroupedSequence>` gathers the values of the
last variable into bins, so one key covers a range of them. ``size`` is how
many values a bin spans, ``start`` is the value the bins are aligned on, and
the last pair of the key carries the lower bound of the bin.

``dtype`` declares the types used to store partition keys, which can differ
from the input variable types. Each lower bound must fit its declared key
type, otherwise splitting raises ``ValueError``. Use a signed key type when
the alignment produces negative bounds, even with unsigned input values.

.. jupyter-execute::

    for start in (1, 0):
        scheme = partitioning.GroupedSequence(('cycle', 'pass'),
                                              size=3,
                                              start=start)
        keys = [item for item, _ in scheme.split_dataset(passes, 'n')]
        bins = sorted({item[1] for item in keys})
        print(f'start={start}: {len(keys)} partitions, bins {bins}')

What the keys are for
---------------------

The names a key carries are the vocabulary of the ``filters`` expression
that selects partitions to read or update:

.. code-block:: python

    collection.load(filters='year == 2022 and month == 6')
    collection.load(filters='cycle == 2 and pass > 40')

A collection partitioned by year has no ``month`` to filter on. The resolution
therefore sets how finely an operation can be aimed, as well as how many
partitions there are.

Choosing a resolution
---------------------

The finer the resolution, the less data an operation reads and the more
partitions there are. One year of hourly measurements — 8760 points, one
``float32`` variable, 105 120 bytes of data — written at three resolutions:

=============  ==========  ====================  =========================
Resolution     Partitions  Metadata, in bytes    Points read for one day
=============  ==========  ====================  =========================
``Y``                   1                 4 285                     8 760
``M``                  12                38 201                       720
``D``                 365             1 123 933                        24
=============  ==========  ====================  =========================

The data is the same 105 120 bytes in the three collections. At a daily
resolution it is 9% of what the collection occupies, and the rest is the
``zarr.json`` of each partition and of each of its variables.

Choose the resolution at which one operation works on the data of one
partition.

A sequence partitioning poses the same question through the number of
distinct keys. ``GroupedSequence`` lowers that number by binning the last
variable, and the collection stays partitioned on the same variables.
