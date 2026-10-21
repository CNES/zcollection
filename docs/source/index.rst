ZCollection
===========

ZCollection stores a dataset too large for one file as a **collection**: a
tree of partitions, each holding a slice of the dataset in the
`Zarr <https://zarr.readthedocs.io/en/stable/>`_ format. The collection is
read and written as a single
:py:class:`dataset <zcollection.dataset.Dataset>`.

A month of new measurements inserted into a ten-year collection is written to one
or two of its 120 partitions.

Concepts
--------

Partition
    A slice of the dataset, written as a Zarr group. The
    :py:mod:`partitioning <zcollection.partitioning>` sets which measurements a
    partition holds and what it is called:
    :py:class:`by date <zcollection.partitioning.date.Date>`,
    ``year=2022/month=01``;
    :py:class:`by sequence <zcollection.partitioning.sequence.Sequence>` of
    variables, ``cycle=2/pass=42``.

Immutable variable
    A variable that does not span the dimension the collection is partitioned
    along. Its values are the same in every partition, so the collection
    stores it once, in ``.immutable``.

Merge strategy
    The rule applied when an insertion lands on a partition that already
    holds data. The new data replaces it, unless the collection is given a
    :py:mod:`merge callable <zcollection.merging>` that combines the two;
    :py:func:`merge_time_series
    <zcollection.merging.time_series.merge_time_series>` is the one the
    package provides.

View
    Variables computed from a collection and stored beside it. A
    :py:class:`view <zcollection.view.View>` opens the collection it
    references read only, and holds its own variables under the same
    partition names.

Layout
------

A collection partitioned by month, holding a longitude that is the same in
every partition: ::

   collection/
   ├── .zcollection
   ├── .immutable/
   │    ├── lon/
   │    │    ├── c/
   │    │    │    └── 0
   │    │    └── zarr.json
   │    └── zarr.json
   └── year=2022/
        ├── month=01/
        │    ├── time/
        │    │    ├── c/
        │    │    │    └── 0
        │    │    └── zarr.json
        │    ├── var1/
        │    │    ├── c/
        │    │    │    └── 0/
        │    │    │         └── 0
        │    │    └── zarr.json
        │    └── zarr.json
        └── month=02/
             └── ...

``.zcollection`` declares the variables, their data types and the
partitioning. Each ``zarr.json`` holds the metadata of the group or of the
array it sits in, and ``c`` holds the chunks of an array.

Scaling
-------

Every method that acts on the partitions takes a ``distributed`` parameter.
It defaults to ``True``, which spreads the work over a `Dask
<https://dask.org/>`_ cluster. With ``distributed=False`` the work stays in
the calling process, which suits a collection small enough for one machine.

What a read gives back is a separate choice. With ``delayed=True`` the
variables are filled with `Dask arrays
<https://docs.dask.org/en/latest/array.html>`_, which read their values when
a computation asks for them. With ``delayed=False`` they are filled with
NumPy arrays, read there and then.

File systems
------------

A collection is stored wherever the `fsspec
<https://filesystem-spec.readthedocs.io/en/latest/>`_ library reaches: a
local disk, an S3 bucket, and the other file systems it implements. The
tests cover a local disk and S3.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   getting_started
   partitioning
   auto_examples/index.rst
   api
   release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
