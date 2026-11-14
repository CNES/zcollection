ZCollection
===========

ZCollection stores large datasets as **collections** of partitions, each
containing a subset of the data in `Zarr <https://zarr.readthedocs.io/en/stable/>`_
format. Although the data is split across multiple partitions, the
collection can be read and written as a single
:py:class:`dataset <zcollection.dataset.Dataset>`.

Partitioning makes it possible to update part of a dataset without rewriting
the entire collection. For example, in a collection partitioned by month,
adding a month of measurements to ten years of data requires writing only
one or two of the 120 partitions.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :link: user_guide
      :link-type: doc
      :padding: 2

      .. card-header::

         **User guide**

      Learn how to create, manage, and work with collections, step by step.

   .. grid-item-card::
      :link: auto_examples/index
      :link-type: doc
      :padding: 2

      .. card-header::

         **Examples**

      Explore complete examples covering collections, views, indexing, and
      working without a Dask cluster.

   .. grid-item-card::
      :link: api
      :link-type: doc
      :padding: 2

      .. card-header::

         **API reference**

      Explore the library's classes, functions, and their parameters.

   .. grid-item-card::
      :link: release
      :link-type: doc
      :padding: 2

      .. card-header::

         **Release notes**

      See what's new, improved, and changed in each release.


Concepts
--------

Partition
    A subset of the dataset stored as a Zarr group. The
    :py:mod:`partitioning <zcollection.partitioning>` scheme determines how
    measurements are grouped into partitions and how those partitions are
    named.

    Partitions can be organized :py:class:`by date
    <zcollection.partitioning.date.Date>`, such as
    ``year=2022/month=01``, or :py:class:`by sequences of variable values
    <zcollection.partitioning.sequence.Sequence>`, such as
    ``cycle=2/pass=42``.

Partition index
    A record of the partitions known to a collection or a view. See
    :doc:`partition_index` for details on how the index is used and how to
    recover from inconsistencies.

Immutable variable
    A variable that does not span the dimension used to partition the
    collection. Its values are identical across partitions and are stored
    only once, in the ``.immutable`` directory.

Merge strategy
    Determines how new data is handled when it is inserted into an existing
    partition. By default, the new data replaces the existing data.

    A custom :py:mod:`merge function <zcollection.merging>` can instead
    combine the two. ZCollection provides
    :py:func:`merge_time_series
    <zcollection.merging.time_series.merge_time_series>` for this purpose.

View
    A set of variables computed from a collection and stored in a separate
    directory. A :py:class:`view <zcollection.view.View>` references its
    source collection in read-only mode and stores its own variables using
    the same partition names.


Storage layout
--------------

The following example shows a collection partitioned by month, with a
longitude variable shared by all partitions::

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

The ``.zcollection`` file defines the collection's variables, their data
types, and the partitioning scheme. Each ``zarr.json`` file contains the
metadata for its corresponding Zarr group or array.

Array chunks are stored in files named with a ``c`` prefix followed by
dot-separated indices, such as ``c.0`` or ``c.0.0``.


Execution
---------

By default, operations that accept a ``distributed`` parameter use a
Dask cluster (``distributed=True``). Set ``distributed=False`` to run
them without a cluster.

The ``map`` and ``map_overlap`` methods return a
:py:class:`dask.bag.Bag`, allowing you to choose how the results are
computed. For example, to execute a mapping in the calling process:

.. code-block:: python

    bag = collection.map(lambda ds: ds.dimensions[collection.dimension])
    results = bag.compute(scheduler='synchronous')

See :doc:`auto_examples/ex_without_dask` for examples of reading, updating,
and mapping a collection without a Dask cluster.


Loading data
------------

The ``delayed`` parameter controls when data is read and which type of
arrays are returned:

- ``delayed=True`` returns `Dask arrays
  <https://docs.dask.org/en/latest/array.html>`_, which load their data
  when a computation requires it.
- ``delayed=False`` returns NumPy arrays, with the data loaded immediately.

For ``load`` and ``update``, both on collections and views, the default
is ``delayed=None``, which follows the ``distributed`` setting: Dask
arrays with ``distributed=True`` and NumPy arrays with
``distributed=False``.

For ``map`` and ``map_overlap``, ``delayed=True`` is the default.

See :doc:`computing` and :doc:`updating` for more information about
function arguments, overlap windows, and execution options.


File systems
------------

ZCollection uses `fsspec
<https://filesystem-spec.readthedocs.io/en/latest/>`_ to access storage
backends, including local file systems, S3, and other file systems supported
by the library.

The test suite covers both local file systems and S3.


.. toctree::
   :hidden:

   user_guide
   auto_examples/index.rst
   api
   release
