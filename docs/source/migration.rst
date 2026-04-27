Migrating to the v3 rewrite
===========================

zcollection releases use a date-based version scheme (``YYYY.M.D``).
The **v1 line** ends at ``2024.2.0`` and is built on Zarr v2; the
**v3 rewrite** starts at ``2026.4.0`` (alpha pre-releases tagged
``2026.4.0aN``) and is built on Zarr v3. The version number alone
doesn't communicate the break — both look like routine date
releases — so the rewrite is referred to in the docs and changelogs
as **the v3 line** (after its on-disk Zarr format), and users on the
v1 line are expected to pin accordingly.

The v3 line is a clean break from v1: the on-disk format is now
`Zarr v3 <https://zarr.readthedocs.io/>`_, the public API has been
redesigned around an immutable schema, and the storage backend is
chosen by URL scheme. There is no Zarr-v2 read path — v1 collections
must be rewritten through the migration tool, not opened in place.

Staying on v1
-------------

If you cannot upgrade yet:

- **Pin** ``zcollection >= 2024.2.0, < 2026.4.0`` in your
  requirements (or the equivalent ``zcollection ~= 2024.2.0`` if you
  want to stay locked to the ``2024.2.x`` patch line).
- The v1 source lives on the ``support/v1`` branch (Zarr/CPython
  convention: ``support/<line>`` for maintenance branches). Critical
  fixes are backported there as ``2024.2.x`` patch tags; no new
  features land.
- The **v1 documentation** is published on Read the Docs as the
  ``support-v1`` (or ``2024.2.0``) version — pick it from the
  version selector in the docs sidebar. The default landing page
  (``stable``) tracks the v3 line.

What changed at a glance
------------------------

The bullets below contrast the v1 line (last release ``2024.2.0``)
with the v3 line (``2026.4.0`` and later).

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
* **Hierarchical datasets**: a :py:class:`~zcollection.Dataset` is now a
  root :py:class:`~zcollection.Group` and can declare nested child groups
  (e.g. ``/data_01/ku/...``). Each group becomes a real Zarr v3
  subgroup on disk; nested groups round-trip through partition I/O. The
  schema gains :py:class:`~zcollection.GroupSchema` and a ``group=``
  keyword on the builder methods.

Entry points
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - v1 (≤ ``2024.2.0``)
     - v3 (``2026.4.0`` and later)
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

The following v1 features have no equivalent in the v3 line:

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
  ``zcollection-migrate`` tool to rewrite a v1 layout into the v3
  on-disk format.
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

The v1 ``Indexer`` was an abstract class with ``_create`` / ``_update``
hooks. In the v3 line :py:class:`~zcollection.indexing.Indexer` is a
single concrete Parquet-backed class that takes a builder callback:

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

When the schema declares nested groups (see
:doc:`auto_examples/ex_groups`), each group materialises as a real
Zarr v3 subgroup inside every partition::

   collection/
   └── year=2024/month=03/
       ├── zarr.json
       ├── time/{zarr.json,c/0}
       └── data_01/
           ├── zarr.json
           └── ku/
               ├── zarr.json
               └── power/{zarr.json,c/0/0}
