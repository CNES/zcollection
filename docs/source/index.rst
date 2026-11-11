ZCollection
===========

ZCollection stores a dataset too large for one file as a **collection**: a
tree of partitions, each holding a slice of the dataset in the
`Zarr <https://zarr.readthedocs.io/en/stable/>`_ format. The collection is
read and written as a single
:py:class:`dataset <zcollection.dataset.Dataset>`.

With monthly date partitioning, a month of new measurements inserted into a
ten-year collection is written to one or two of its 120 partitions.

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

Partition index
    The list of partitions known to a collection or a view.
    :doc:`partition_index` explains its role and how to recover from
    index problems.

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
   ├── .partitions
   ├── .immutable/
   │    ├── lon/
   │    │    ├── c.0
   │    │    └── zarr.json
   │    └── zarr.json
   └── year=2022/
        ├── month=01/
        │    ├── time/
        │    │    ├── c.0
        │    │    └── zarr.json
        │    ├── var1/
        │    │    ├── c.0.0
        │    │    └── zarr.json
        │    └── zarr.json
        └── month=02/
             └── ...

``.zcollection`` declares the variables, their data types and the
partitioning. Each ``zarr.json`` holds the metadata of the group or of the
array it sits in. Chunk filenames consist of the ``c`` prefix and the
indices, joined with dots.

Scaling
-------

``insert``, ``load``, ``update`` and ``copy`` can be used without a `Dask
<https://dask.org/>`_ cluster by passing ``distributed=False``. The default,
``True``, uses a cluster.

``map`` and ``map_overlap`` return a :py:class:`dask.bag.Bag`. You choose
how to compute the results when you call ``compute()``. For an existing
collection, compute a mapping in the calling process with:

.. code-block:: python

    bag = collection.map(lambda ds: ds.dimensions[collection.dimension])
    results = bag.compute(scheduler='synchronous')

See :doc:`auto_examples/ex_without_dask` for reading, updating and mapping
without a cluster.

What a read gives back is a separate choice. With ``delayed=True`` the
variables are filled with `Dask arrays
<https://docs.dask.org/en/latest/array.html>`_, which read their values when
a computation asks for them. With ``delayed=False`` they are filled with
NumPy arrays, read there and then.

See :doc:`computing` and :doc:`updating` for function arguments, overlap
windows and execution choices.

File systems
------------

A collection is stored wherever the `fsspec
<https://filesystem-spec.readthedocs.io/en/latest/>`_ library reaches: a
local disk, an S3 bucket, and the other file systems it implements. The
tests cover a local disk and S3.

.. toctree::
   :maxdepth: 2
   :caption: First steps

   install
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User guide

   partitioning
   inserting
   computing
   updating
   managing
   failures
   partition_index
   migration

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
