Migrating from 1.x to 2.0
=========================

zcollection ``2.0`` is a clean break from the ``1.x`` line. The on-disk
format is now `Zarr v3 <https://zarr.readthedocs.io/>`_, the public API has
been redesigned around an immutable schema, and the storage backend is
chosen by URL scheme. There is no v2 read path — ``1.x`` collections must
be rewritten through the migration tool, not opened in place.

If you cannot upgrade yet, the ``1.x`` line is preserved on the
``legacy/v2`` branch.

What changed at a glance
------------------------

* **Storage format**: Zarr v3 (sharded, async-friendly) instead of Zarr v2.
  The ``.zarray`` / ``.zgroup`` / ``.zattrs`` / ``.zmetadata`` files are
  gone; each partition is a single Zarr v3 group with a ``zarr.json``.
* **Schema is explicit**: build a :py:class:`~zcollection.Schema` once and
  pass it to :py:func:`~zcollection.create_collection`. No more inferring
  metadata from the first dataset.
* **Stores by URL scheme**: ``file://``, ``memory://``, ``s3://``,
  ``icechunk://`` are dispatched to the right backend by
  :py:func:`~zcollection.open_store`.
* **Async core, sync facade**: the implementation is async; the sync API
  in :py:mod:`zcollection` is a thin wrapper. An
  :py:mod:`zcollection.aio` mirror is published as stable.
* **Transactions**: :py:class:`~zcollection.store.IcechunkStore` makes
  ``insert`` / ``drop_partitions`` atomic. Without Icechunk, multi-process
  writers are not supported.
* **Partition catalog**: the optional ``_catalog/`` group lists partitions
  in O(1) and replaces the per-partition ``LIST`` walk on cloud stores.

Entry points
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 1.x
     - 2.0
   * - ``zcollection.create_collection(axis, ds, partitioner, path, filesystem=fs)``
     - ``zcollection.create_collection(url, *, schema, axis, partitioning)``
   * - ``zcollection.open_collection(path, filesystem=fs)``
     - ``zcollection.open_collection(url, *, mode='r'|'rw')``
   * - ``zcollection.create_view(path, view_ref, ds, filesystem=fs)``
     - ``zcollection.view.View.create(store, base=, variables=, reference=)``
   * - ``zcollection.open_view(path, filesystem=fs)``
     - ``zcollection.view.View.open(store, base=)``

Removed
-------

The following ``1.x`` features have no equivalent in ``2.0``:

* ``synchronizer=`` — replaced by per-partition ``asyncio.Lock``,
  Dask DAG topology, and (for cross-partition atomicity)
  :py:class:`~zcollection.store.IcechunkStore`.
* ``delayed=`` flag at the dataset level — each
  :py:class:`~zcollection.Variable` is polymorphic on its data
  (numpy / dask / Zarr ``AsyncArray``).
* ``distributed=`` and ``filesystem=`` keyword arguments — pass a Dask
  client directly when you want one; pass a URL or pre-opened
  :py:class:`~zcollection.store.Store` for the backend.
* ``partition_base_dir`` positional — encode it in the URL.
* ``update_deprecated_collection`` — moved out of the package; use the
  ``zcollection-migrate`` tool to rewrite a 1.x layout to 2.0.
* ``Dataset.load`` / ``Dataset.add_variable`` / ``Dataset.drop_variable`` /
  ``Dataset.metadata`` — schemas are immutable; build a new
  :py:class:`~zcollection.Schema` instead.
* The ``-1`` chunk sentinel — use ``chunks=None`` for unknown.
* The ``filler: bool`` flag on variables — use
  :py:class:`~zcollection.schema.VariableRole`.

Renamed
-------

* ``zcollection.merging`` → :py:mod:`zcollection.collection.merge`
* ``merge_callable=`` keyword → ``merge=``
* ``zcollection.dataset`` → :py:mod:`zcollection.data`
* ``zcollection.variable.array`` / ``delayed_array`` → single
  :py:class:`zcollection.Variable`
* ``zcollection.fs_utils`` → :py:mod:`zcollection.store`
* ``zcollection.dask_utils`` → :py:mod:`zcollection.dask`
* ``zcollection.expression`` → :py:mod:`zcollection.partitioning.expression`
* ``zcollection.meta`` → :py:mod:`zcollection.schema`

Code translation
----------------

Creating a collection
~~~~~~~~~~~~~~~~~~~~~

Before:

.. code-block:: python

   import fsspec
   import zcollection
   from zcollection.partitioning import Date

   fs = fsspec.filesystem('file')
   col = zcollection.create_collection(
       'time', ds, Date(('time',), 'M'),
       '/data/altimetry', filesystem=fs,
   )

After:

.. code-block:: python

   import zcollection as zc
   from zcollection.partitioning import Date

   schema = (
       zc.Schema()
       .with_dimension('time', chunks=4096)
       .with_dimension('x_ac', size=240, chunks=240)
       .with_variable('time', dtype='int64', dimensions=('time',))
       .with_variable('ssh', dtype='float32',
                      dimensions=('time', 'x_ac'))
       .build()
   )
   col = zc.create_collection(
       'file:///data/altimetry',
       schema=schema, axis='time',
       partitioning=Date(('time',), resolution='M'),
   )
   col.insert(ds)

Views
~~~~~

Before:

.. code-block:: python

   from zcollection.view import ViewReference
   view = zcollection.create_view(
       '/data/view', ViewReference('/data/altimetry'),
       ds, filesystem=fs,
   )

After:

.. code-block:: python

   from zcollection.view import View, ViewReference
   view_var = zc.VariableSchema(
       name='var2', dtype='float32',
       dimensions=('time', 'x_ac'),
   )
   view = View.create(
       zc.open_store('file:///data/view'),
       base=base,
       variables=[view_var],
       reference=ViewReference(uri='file:///data/altimetry'),
   )

Indexing
~~~~~~~~

The 1.x ``Indexer`` was an abstract class with ``_create`` / ``_update``
hooks. In 2.0, :py:class:`~zcollection.indexing.Indexer` is a single
concrete Parquet-backed class that takes a builder callback:

.. code-block:: python

   from zcollection.indexing import Indexer

   def builder(ds):
       # return a structured numpy array with at least
       # `_start`, `_stop`, and the key columns
       ...

   indexer = Indexer.build(collection, builder=builder)
   indexer.write('/data/index.parquet')
   ranges = indexer.lookup(pass_number=[1, 2])

Atomic writes
~~~~~~~~~~~~~

To get crash-safe inserts, open the collection through
:py:class:`~zcollection.store.IcechunkStore`:

.. code-block:: python

   col = zc.create_collection(
       'icechunk:///data/altimetry',
       schema=schema, axis='time',
       partitioning=Date(('time',), 'M'),
   )

A failed ``insert`` is rolled back to the prior commit; no partial state
is ever visible after reopen.

Layout on disk
--------------

Old (Zarr v2)::

   collection/
   ├── year=2024/month=03/
   │   ├── time/{0.0,.zarray,.zattrs}
   │   ├── ssh/{0.0,.zarray,.zattrs}
   │   └── {.zattrs,.zgroup,.zmetadata}
   └── .zcollection

New (Zarr v3)::

   collection/
   ├── zarr.json
   ├── _zcollection.json
   ├── _catalog/{zarr.json,c/0}        # optional partition index
   ├── _immutable/{zarr.json,...}      # non-partitioned variables
   └── year=2024/month=03/
       ├── zarr.json
       ├── time/{zarr.json,c/0}
       └── ssh/{zarr.json,c/0/0}
