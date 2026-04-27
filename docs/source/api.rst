:tocdepth: 2

API Reference
#############

This page documents the public surface of zcollection. Everything listed
here is importable from the top-level :mod:`zcollection` package (with a
few sub-namespaces for grouped APIs: :mod:`zcollection.codecs`,
:mod:`zcollection.merge`, :mod:`zcollection.partitioning`,
:mod:`zcollection.view`, :mod:`zcollection.aio`).

Building a schema
=================

A :class:`~zcollection.DatasetSchema` is the immutable description of a
dataset: dimensions, variables, attributes, child groups, and a format
version. The recommended way to build one is the fluent
:func:`~zcollection.Schema` factory:

.. code-block:: python

   import numpy
   import zcollection as zc

   schema = (
       zc.Schema()
       .with_dimension("time", chunks=4096)
       .with_dimension("x_ac", size=240, chunks=240)
       .with_variable("time", dtype="datetime64[ns]", dimensions=("time",))
       .with_variable(
           "ssh",
           dtype="float32",
           dimensions=("time", "x_ac"),
           fill_value=numpy.float32("nan"),
           codecs=zc.codecs.profile("cloud-balanced"),
       )
       .build()
   )

Hierarchical schemas
--------------------

Variables can be placed inside nested groups by passing ``group=`` to
:meth:`~zcollection.SchemaBuilder.with_dimension`,
:meth:`~zcollection.SchemaBuilder.with_variable`, and
:meth:`~zcollection.SchemaBuilder.with_attribute`. Use
:meth:`~zcollection.SchemaBuilder.with_group` to attach group-level
attributes ahead of time. Intermediate groups along the path are
created on demand. Variables in a child group may reference dimensions
declared on any ancestor (dimension inheritance):

.. code-block:: python

   schema = (
       zc.Schema()
       .with_dimension("time", chunks=4096)
       .with_variable("time", dtype="int64", dimensions=("time",))
       .with_group("/data_01/ku", attrs={"band": "Ku"})
       .with_dimension("range", size=240, chunks=240, group="/data_01/ku")
       .with_variable(
           "power",
           dtype="float32",
           dimensions=("time", "range"),  # ``time`` inherited from root
           group="/data_01/ku",
       )
       .build()
   )

The resulting :class:`~zcollection.DatasetSchema` exposes the hierarchy
through :attr:`~zcollection.GroupSchema.groups` and
:meth:`~zcollection.GroupSchema.all_variables` (keys are absolute paths).
:meth:`~zcollection.DatasetSchema.with_partition_axis` recurses into
nested groups, marking variables that don't span the partitioning axis
as immutable.

.. autofunction:: zcollection.Schema

.. autoclass:: zcollection.SchemaBuilder
   :members:

.. autoclass:: zcollection.DatasetSchema
   :members:

.. autoclass:: zcollection.GroupSchema
   :members:

.. autoclass:: zcollection.Dimension
   :members:

.. autoclass:: zcollection.VariableSchema
   :members:

.. autoclass:: zcollection.VariableRole
   :members:


Creating and opening collections
================================

A :class:`~zcollection.Collection` is the persistent, partitioned
container. The two entry points dispatch on URL scheme
(``file://``, ``memory://``, ``s3://``, ``icechunk://``):

.. autofunction:: zcollection.create_collection

.. autofunction:: zcollection.open_collection


Working with a Collection
=========================

.. autoclass:: zcollection.Collection
   :members: store, schema, axis, partitioning, read_only,
             partitions, query, query_async,
             insert, insert_async,
             update, update_async,
             map, map_async,
             drop_partitions,
             repair_catalog


Merge strategies
================

Strategies for inserting into a partition that already has data on disk.
Pass any of these to :meth:`Collection.insert` via the ``merge=``
argument, either as the callable or by name (``"replace"``, ``"concat"``,
``"time_series"``, ``"upsert"``).

.. autofunction:: zcollection.merge.replace

.. autofunction:: zcollection.merge.concat

.. autofunction:: zcollection.merge.time_series

.. autofunction:: zcollection.merge.upsert

.. autofunction:: zcollection.merge.upsert_within

.. autoclass:: zcollection.merge.MergeCallable
   :members:


Partitioning
============

Partitioning strategies decide how rows are bucketed onto disk. Pick one
when you create the collection; it's persisted and used for every
subsequent open.

.. autoclass:: zcollection.partitioning.Date
   :members:
   :inherited-members:

.. autoclass:: zcollection.partitioning.Sequence
   :members:
   :inherited-members:

.. autoclass:: zcollection.partitioning.GroupedSequence
   :members:
   :inherited-members:

.. autoclass:: zcollection.partitioning.Partitioning
   :members:

.. autofunction:: zcollection.partitioning.compile_filter

The :func:`~zcollection.partitioning.compile_filter` helper turns a
filter string (the kind passed to ``Collection.partitions(filters=…)``)
into a typed predicate. You usually do not need to call it directly.


Views
=====

A view overlays extra variables on top of an existing read-only base
collection without copying its data. Views live in a separate store and
share the partitioning of the base.

.. autoclass:: zcollection.view.View
   :members: create, open, store, base, view_schema, reference,
             variables, read_only,
             partitions, query, query_async, update, update_async

.. autoclass:: zcollection.view.ViewReference
   :members:


Indexing
========

A :class:`~zcollection.indexing.parquet.Indexer` is a Parquet-backed
lookup table: it maps user-defined key columns to ``(partition, start,
stop)`` row ranges so callers can slice a collection without scanning
every partition.

.. autoclass:: zcollection.indexing.parquet.Indexer
   :members: build, read, write, table, key_columns, lookup


Datasets, groups, and variables (in memory)
===========================================

The objects you receive from :meth:`Collection.query` and pass back into
:meth:`Collection.insert`. :class:`~zcollection.Dataset` is a root
:class:`~zcollection.Group`: it owns variables and child groups
directly. :class:`~zcollection.Group` itself is the hierarchical
container reused for every nested group.

Both :class:`~zcollection.Dataset` and :class:`~zcollection.Variable`
print as multi-line, xarray-like blocks that include a synthetic byte
size for the dataset, each child group, and each variable, so you can
gauge memory/disk footprint at a glance::

   <zcollection.data.dataset.Dataset '/'> Size: 5.32 MB
     Dimensions: (time: 100000, x_ac: 240)
   Data variables:
       time (time)              int64           781.25 kB  numpy.ndarray<...>
       ssh  (time, x_ac)        float32         91.55 MB   numpy.ndarray<...>
   Groups:
       data_01           1.20 GB  (0 variables, 1 subgroup)

Path-based access (``ds.get_variable("/data_01/ku/power")``,
``ds["/data_01/ku/power"]``) and short-name search
(:meth:`~zcollection.Group.find_variable`,
:meth:`~zcollection.Group.find_group`) make navigating the hierarchy
straightforward. :meth:`~zcollection.Group.find_dimension` walks up the
tree so child groups inherit dimensions declared on ancestors.
:attr:`~zcollection.Group.nbytes` recurses over the whole tree.

The xarray bridge (:meth:`~zcollection.Dataset.to_xarray` /
:meth:`~zcollection.Dataset.from_xarray`) operates on the *root* group
only — xarray has no native group concept, so nested groups are dropped
on conversion.

.. autoclass:: zcollection.Dataset
   :members:

.. autoclass:: zcollection.Group
   :members:

.. autoclass:: zcollection.Variable
   :members:


Codecs
======

Codecs describe how a variable's chunks travel from numpy memory to
bytes on disk. Most users only need named profiles:

.. code-block:: python

   import zcollection as zc

   stack = zc.codecs.profile("cloud-balanced")
   names = zc.codecs.profile_names()  # ('local-fast', 'cloud-balanced', ...)

.. autofunction:: zcollection.codecs.profile

.. autofunction:: zcollection.codecs.profile_names

.. autofunction:: zcollection.codecs.auto_codecs

.. autoclass:: zcollection.codecs.CodecStack
   :members:

.. data:: zcollection.codecs.DEFAULT_PROFILE

   Name of the profile used when no ``codecs=`` is given on a variable.


Stores
======

Stores are the I/O backend. The factory selects an implementation from a
URL — that is what :func:`create_collection` and :func:`open_collection`
call internally — but you can also build one explicitly and pass it as
the ``path`` argument.

.. autofunction:: zcollection.open_store

.. autoclass:: zcollection.Store
   :members:

.. autoclass:: zcollection.LocalStore

.. autoclass:: zcollection.MemoryStore

The optional :class:`zcollection.store.IcechunkStore` is loaded lazily
when ``icechunk://`` is opened or when you ``import`` it explicitly. It
unlocks transactional, multi-writer inserts.


Errors
======

All exceptions raised by zcollection inherit from
:class:`~zcollection.ZCollectionError`.

.. autoexception:: zcollection.ZCollectionError
.. autoexception:: zcollection.SchemaError
.. autoexception:: zcollection.StoreError
.. autoexception:: zcollection.CollectionExistsError
.. autoexception:: zcollection.CollectionNotFoundError
.. autoexception:: zcollection.ReadOnlyError


Async API
=========

:mod:`zcollection.aio` mirrors the sync facade one-to-one. Every
function returns a coroutine; every method has an ``_async`` counterpart
on :class:`Collection` and :class:`view.View`. Use this module when you
are already inside an event loop.

.. automodule:: zcollection.aio
   :members:
