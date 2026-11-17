Migrating a collection to Zarr 3
================================

ZCollection provides tools to migrate collections created by earlier
releases from Zarr 2 to Zarr 3.

Migration creates a new collection without modifying the original.
You can migrate partitions in batches, verify their contents, and
remove the source partitions once they have been migrated successfully.

This allows large collections to be migrated progressively, without
requiring enough additional storage to hold both collections in full.


Reading an older collection
---------------------------

Collections created by earlier releases use the Zarr 2 format. They
remain readable, but cannot be modified with the current release.

Opening an older collection succeeds with a warning:

.. code-block:: python

    import zcollection

    collection = zcollection.open_collection("/data/mycollection")

.. code-block:: text

    UserWarning: This collection was written by an earlier release and can
    only be read. Use the 'zcollection.migration.migrate' function to copy it into a
    collection this release writes.

The collection retains its data, masks, and attributes, and all read
operations remain available.

Writing is not supported:

- Opening the collection for writing raises ``ValueError``.
- Operations such as inserting data or adding a variable raise
  :py:class:`io.UnsupportedOperation`.

To modify the collection with the current release, migrate it to
Zarr 3 first.


Migrating a collection
----------------------

:py:func:`zcollection.migration.migrate` reads an existing collection
and writes its contents to a new location. The source collection is
never modified by the migration.

The migration utilities are available through ``zcollection.migration``
and do not need to be imported by applications that only read
collections.

To migrate a collection:

.. code-block:: python

    from zcollection import migration

    outcome = migration.migrate("/data/mycollection",
                                "/data/mycollection.v3")

The first call creates the target collection using the configuration
of the source.

Each partition is read, decoded, and written again in Zarr 3 format.
The migration preserves:

- Variable dimensions, attributes, and fill values.
- Compressors and filters.
- The original chunking.
- The values and masks returned when reading each variable.

Immutable variables are migrated once, under ``.immutable``.

Before copying data, ``migrate`` and ``migrate_view`` inspect the source
schema and log a message at ``WARNING`` level listing variables whose data
types use storage formats that are not yet standardized in Zarr V3, as
reported by the installed Zarr version. The message includes variable names
and NumPy dtypes, including immutable variables for collections, without
reading array values. For example, ``S1`` and ``S7`` identify fixed-length
byte strings. Migration proceeds without changing the source collection.
However, future Zarr versions or other Zarr libraries may not be able to
read these variables from the migrated collection. Zarr may also emit its
own warnings during the copy.

The source and target can use different file systems, specified through
``source_filesystem`` and ``target_filesystem``.

For example, to migrate a local collection to S3:

.. code-block:: python

    outcome = migration.migrate("/data/mycollection",
                                "bucket/mycollection",
                                target_filesystem="s3")


What changes on disk
~~~~~~~~~~~~~~~~~~~~

Although the data and metadata exposed by ZCollection are preserved,
the physical representation changes during migration:

- Chunks are compressed again.
- Chunks containing only the array's fill value are omitted.
- Chunk filenames use a ``c`` prefix followed by dot-separated indices.

Applications reading through ZCollection continue to see the same
values and masks. Applications accessing the stored files directly
must account for the new Zarr 3 layout.

The migration only copies data described by the collection model.
Additional files stored at the collection's root, such as custom
indexes or catalogues, are not copied.

These files must be transferred separately.


Migrating in batches
--------------------

Large collections can be migrated a few partitions at a time. The
source collection does not need to remain frozen throughout the
entire migration.

The ``migrate``, ``check``, and ``prune`` functions support the same
partition selection parameters:

- ``filters`` selects partitions using their partition keys, either
  through an expression or a callable.
- ``selected_partitions`` selects partitions by their relative paths.

When both are provided, only partitions matching both selections
are processed.

For example:

.. code-block:: python

    migration.migrate(source, target, filters="year == 2014 and month == 5")
    migration.migrate(source, target, filters=lambda keys: keys["year"] < 2016)
    migration.migrate(source, target,
                      selected_partitions=["year=2014/month=05/day=24"])

Without either parameter, ``migrate`` processes every partition
not already present and complete in the target.


Resuming an interrupted migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Migration can be safely resumed by calling ``migrate`` again with the
same source and target.

Partitions already complete in the target are skipped. To identify them,
a subsequent call lists the source partitions and reads the metadata of
their corresponding partitions in the target.

A partition left partially written has no consolidated metadata.
It is therefore considered missing and is rewritten in full by the
next migration attempt.

No special recovery procedure is required after an interruption:
simply run ``migrate`` again.

You can migrate, check, and prune successive batches independently.
Processing a batch through all three steps before moving to the next
keeps the additional storage requirements close to the size of the
batch rather than the entire collection.


Monitoring progress
-------------------

:py:func:`zcollection.migration.status` compares the partitions of
the source and target collections. It groups the results by the first
key of the partitioning scheme.

.. code-block:: python

    print(migration.status("/data/mycollection", "/data/mycollection.v3"))

.. code-block:: text

    year=2014 : 365 migrated, 0 missing
    year=2015 : 212 migrated, 153 missing
    year=2016 : 0 migrated, 366 missing

The status distinguishes four states:

- **migrated:** The partition exists in the source and is complete
  in the target.
- **missing:** The partition exists in the source but is not complete
  in the target.
- **pruned:** The partition has been removed from the source and
  remains in the target.
- **lost:** The partition is no longer present in the source and is
  not complete in the target.

The ``pruned`` and ``lost`` counts are displayed only when nonzero.

A lost partition cannot be recovered from either collection. This
makes it important to verify migrated data before removing the source.


Verifying migrated data
-----------------------

:py:func:`zcollection.migration.check` compares the source and target
collections to verify that the migration preserved their contents.

It reads every value of every selected variable on both sides and
compares:

- Data values and masks.
- Variable attributes and fill values.
- Dimensions and chunking.

The check must be called explicitly; migration does not run it
automatically.

For example, to verify the partitions of 2014:

.. code-block:: python

    outcome = migration.check("/data/mycollection",
                              "/data/mycollection.v3",
                              filters="year == 2014")

    assert not outcome.failed

The result contains three attributes:

- ``outcome.done`` lists partitions whose values match.
- ``outcome.skipped`` lists partitions that are not complete in the
  target.
- ``outcome.failed`` identifies partitions that differ, along with
  the names of the variables containing differences.

You can run the check as often as needed.

Verification reads the selected data from both collections, making
it comparable in cost to the migration itself. Checking individual
batches before pruning them avoids repeatedly reading the entire
collection.


Detecting missing chunks
~~~~~~~~~~~~~~~~~~~~~~~~

During verification, ``check`` also examines the source arrays for
missing chunks.

If an array's metadata indicates more chunks than are physically
present, it logs a warning identifying the affected array.

.. warning::

   Zarr reads missing chunks as fill values. If a chunk is already
   missing from the source, reading both collections may therefore
   produce identical values even though data has been lost.

   Comparing values alone cannot detect this situation. Pay attention
   to warnings about missing source chunks when verifying a migration.


Removing migrated partitions
----------------------------

Once a batch has been migrated and verified, use
:py:func:`zcollection.migration.prune` to remove its partitions from
the source collection.

.. code-block:: python

    outcome = migration.prune("/data/mycollection",
                              "/data/mycollection.v3",
                              filters="year == 2014")

``prune`` only removes source partitions that are complete in the
target.

Partitions that are not complete in the target are preserved in the
source and reported in ``outcome.failed``.

The operation also verifies that the target corresponds to the source.
If the target refers to another collection, ``prune`` raises
``ValueError``.

Only ``prune`` removes data from the source collection. It preserves
the collection's configuration, its immutable variables, and any
additional files stored alongside them.


Distributed execution
---------------------

By default, migration uses a Dask cluster (``distributed=True``).
Each partition is processed in a separate task on the cluster associated
with the current Dask client.

The ``npartitions`` parameter controls how many partition tasks can run
concurrently. Each operation initially submits up to ``npartitions`` tasks,
then submits another whenever one finishes.

By default, ``npartitions=None`` uses the number of Dask workers. This
parameter is available for ``migrate``, ``check``, and ``prune``, as well
as their view equivalents: ``migrate_view``, ``check_view``, and
``prune_view``.

For example, to process up to two partitions concurrently:

.. code-block:: python

    outcome = migration.migrate(source, target, npartitions=2)
    outcome = migration.migrate_view(view_source, view_target, npartitions=2)

Set ``distributed=False`` to execute the operation in the calling
process instead. Partitions then run sequentially regardless of
``npartitions``:

.. code-block:: python

    outcome = migration.migrate(source, target, distributed=False)

A failure in one partition does not stop the migration of the others.
The failed partition is recorded in ``outcome.failed``, together with
its error as text. For partition operations, this text includes a traceback.
A failed comparison names the differing variables in a ``RuntimeError``
message within that traceback.

Call ``migrate`` again to retry partitions that did not complete.

An outcome records the results reported by individual tasks, which
may not always reflect the final state of storage.

For example, Dask can rerun a task when the worker holding its result
is lost. A partition successfully written by an earlier attempt may
then appear under ``skipped`` in the outcome of the retry.

Use :py:func:`zcollection.migration.status` to inspect the partitions
present in storage and :py:func:`zcollection.migration.check` to verify
their contents.


Migrating views
---------------

Views created by earlier releases also remain readable but cannot be
modified with the current release.

Opening an older view produces a warning referring to ``migrate_view``.
If its reference collection also uses the old format, opening the
view produces two warnings, one for each.

The following operations are unavailable on an older view:

- ``add_variable``
- ``drop_variable``
- ``reindex``
- ``sync``
- ``update``

They raise :py:class:`io.UnsupportedOperation`.

``validate_partitions`` can still identify invalid partitions.
However, calling it with ``fix=True`` raises
:py:class:`PermissionError` on the first invalid partition without
modifying the view.


Migrating a view
~~~~~~~~~~~~~~~~

Migrate the reference collection before migrating its views.

When migrating a view, use the ``reference`` parameter to point the
new view to the migrated collection:

.. code-block:: python

    outcome = migration.migrate_view("/data/myview",
                                     "/data/myview.v3",
                                     reference="/data/mycollection.v3")

The ``reference`` path is a path on the target's file system. If
omitted, the migrated view retains the reference collection used by
the original view.

:py:func:`zcollection.migration.migrate_view`,
:py:func:`zcollection.migration.check_view`, and
:py:func:`zcollection.migration.prune_view` support ``filters`` and
``selected_partitions``, using the partitioning scheme of the referenced
collection.

The view migration API also provides the equivalent operations for
tracking, checking, and pruning:

- :py:func:`zcollection.migration.view_status` reports migration progress.
- :py:func:`zcollection.migration.check_view` verifies migrated data.
- :py:func:`zcollection.migration.prune_view` removes migrated source
  partitions.

These correspond to :py:func:`zcollection.migration.status`,
:py:func:`zcollection.migration.check`, and
:py:func:`zcollection.migration.prune` for collections.

A partition that remains in the migrated view but has been removed
from the source view is counted as *pruned*.


Switching to the migrated collection
------------------------------------

Once all partitions have been migrated and verified, the new collection
is ready to replace the original.

The final step depends on the storage system.


Local file systems
~~~~~~~~~~~~~~~~~~

On a file system that supports directory renaming, you can move the
original collection aside and rename the migrated collection to its
original path:

.. code-block:: text

    mv /data/mycollection /data/mycollection.v2
    mv /data/mycollection.v3 /data/mycollection

Readers can then continue using the original collection path.


Object stores
~~~~~~~~~~~~~

Object stores do not support directory renaming in the same way.
Replacing the original path would require copying all the objects
again.

Instead, point readers directly to the migrated collection's new
location. This avoids another complete transfer of the data.


Additional files
~~~~~~~~~~~~~~~~

Remember that migration does not copy files outside the collection
model, such as custom indexes or catalogues stored at its root.

These files must be handled separately.

When renaming a local collection, any such files remain with the
original directory, now moved to the backup path.
