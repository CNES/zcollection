Migrating a collection to Zarr 3
================================

On this page you take a collection written by an earlier release of
zcollection and migrate it into one this release writes, some partitions at
a time, checking each one before you free the space the old collection
takes.

What an old collection allows
-----------------------------

An earlier release wrote its collections in the Zarr 2 format. Opening one
succeeds, with a warning:

.. code-block:: python

    import zcollection

    collection = zcollection.open_collection("/data/mycollection")

.. code-block:: text

    UserWarning: This collection was written by an earlier release and can
    only be read. Use the 'zcollection.migrate' function to copy it into a
    collection this release writes.

Every read works: the partitions keep their values, their masks and their
attributes. Writing is what is refused. Opening the collection for writing
stops with a ``ValueError`` naming the same release, and inserting a dataset
into it or adding a variable to it stops with an
:py:class:`io.UnsupportedOperation`.

A view written by an earlier release does not open at all: reading one stops
with a ``ValueError``, whatever the mode asked for.

Migrating it
------------

:py:func:`zcollection.migrate` reads the source and writes a second
collection. The source is never written.

.. code-block:: python

    outcome = zcollection.migrate("/data/mycollection",
                                  "/data/mycollection.v3")

The first call creates the target and gives it the configuration the source
describes. Each partition is read, decoded and written again, so the
migration carries over the compressor, the filters, the fill value, the dimensions and
the attributes of every variable, and the chunking of the source. The immutable variables are
migrated once, under the name ``.immutable``.

The two collections may live on different file systems, which
``source_filesystem`` and ``target_filesystem`` name:

.. code-block:: python

    outcome = zcollection.migrate("/data/mycollection",
                                  "bucket/mycollection",
                                  target_filesystem="s3")

**The migration carries the data.** A file you keep at the
root of the collection — an index, a catalogue, anything the collection
model does not describe — is left where it is. Copy those yourself.

The values and the masks a variable reads are the ones the source holds.
The bytes on disk are not the same: the chunks are compressed again, a chunk
holding nothing but the fill value of its array is left out, and an array
names the file of a chunk with the ``c`` prefix and the indices joined
with a dot. Nothing downstream that reads through zcollection sees a
difference; anything that reads the files themselves does.

Knowing where the migration has got to
--------------------------------------

:py:func:`zcollection.status` compares the partitions of the two collections
and groups the counts by the first key of the partitioning:

.. code-block:: python

    print(zcollection.status("/data/mycollection", "/data/mycollection.v3"))

.. code-block:: text

    year=2014 : 365 migrated, 0 missing
    year=2015 : 212 migrated, 153 missing
    year=2016 : 0 migrated, 366 missing

A partition both collections hold counts as *migrated* when the target holds
it complete, and as *missing* when it does not. Two counts appear once they are not
empty: *pruned*, the partitions you have already pruned from the source, and
*lost*, the partitions the source no longer holds and the target does not
hold complete. Nothing can rebuild those.

Checking the migration
----------------------

:py:func:`zcollection.check` reads every value of every variable on both
sides and compares them, masks included, along with the attributes, the fill
value, the dimensions and the chunking. It is never called for you, and you
may call it as often as you like.

.. code-block:: python

    outcome = zcollection.check("/data/mycollection",
                                "/data/mycollection.v3",
                                filters="year == 2014")

    assert not outcome.failed

``outcome.done`` names the partitions whose values are equal,
``outcome.skipped`` those the target does not hold complete, and
``outcome.failed`` those that differ, each with the names of the variables
that do.

The check reads the whole of both collections, so it costs about as much as
the migration itself. Running it on the partitions you are about to
prune, rather than on everything, is what keeps that cost in hand.

While it reads the source, the check also logs a warning naming any array whose
metadata calls for more chunks than the source holds.
Zarr reads a chunk it cannot find as the fill value, on both sides, so
comparing the values cannot see such a loss.

Pruning the source
------------------

:py:func:`zcollection.prune` removes from the source the partitions the
target holds complete. It is the only function that removes anything from
the source, and it removes nothing else there: the configuration of the collection, its immutable
variables and the files you keep beside them all stay.

.. code-block:: python

    outcome = zcollection.prune("/data/mycollection",
                                "/data/mycollection.v3",
                                filters="year == 2014")

A partition the target does not hold complete is refused and named in
``outcome.failed``. The call stops with a ``ValueError`` when the target is
not the copy of this source — another collection, or a mistyped path that
happens to name one.

Selecting the partitions to migrate
-----------------------------------

A collection of hundreds of terabytes does not have to be migrated in one
call, and the source does not have to be frozen for the whole of it.
:py:func:`zcollection.migrate`, :py:func:`zcollection.check` and
:py:func:`zcollection.prune` take the same two ways of naming the partitions
to act on, which bound each other:

.. code-block:: python

    zcollection.migrate(source, target, filters="year == 2014 and month == 5")
    zcollection.migrate(source, target, filters=lambda keys: keys["year"] < 2016)
    zcollection.migrate(source, target,
                        selected_partitions=["year=2014/month=05/day=24"])

Naming neither migrates every partition the target does not already hold.

A partition the target holds complete is skipped, so calling
:py:func:`zcollection.migrate` again costs a listing of the source and one
metadata read per partition. That
is also what makes an interrupted migration safe to resume: a partition written
only in part carries no consolidated metadata, counts as missing, and is written
again whole by the next call. Stopping a migration needs no precaution —
stop it, and call it again.

Nothing requires you to migrate, check and prune one set of partitions
before starting the next. Running them in that order, a few partitions at a
time, is what keeps the space the two collections need together to a
fraction more than the source.

With or without Dask
--------------------

The partitions are independent, so they are spread over a Dask cluster by
default, one partition per task. Passing ``distributed=False`` runs them in
the calling process instead:

.. code-block:: python

    outcome = zcollection.migrate(source, target, distributed=False)

A partition the migration stops on does not stop the run: it is named in
``outcome.failed`` with its error as text, and the others are written. Call
:py:func:`zcollection.migrate` again to take them up.

The three fields of an outcome record what each task decided, not what the
two collections hold once the call returns. Dask runs a task again when the
worker holding its result dies, so a partition one call wrote can come back
under ``skipped``. :py:func:`zcollection.status` and
:py:func:`zcollection.check` are what read the storage.

Views
-----

Migrate the collection a view references before the view itself, then point
the copy of the view at the copy of the collection:

.. code-block:: python

    outcome = zcollection.migrate_view("/data/myview",
                                       "/data/myview.v3",
                                       reference="/data/mycollection.v3")

Leaving ``reference`` unset keeps the collection the source view references.
A view is selected the way a collection is:
:py:func:`zcollection.migrate_view` and :py:func:`zcollection.check_view`
take the same ``filters`` and ``selected_partitions``, read against the
partitioning of the collection the view references.

:py:func:`zcollection.view_status` and :py:func:`zcollection.check_view`
answer for a view what :py:func:`zcollection.status` and
:py:func:`zcollection.check` answer for a collection. A partition the copy
keeps and the source view no longer holds is counted under *pruned*; nothing
removes it, there being no counterpart of :py:func:`zcollection.prune` for a
view.

Switching over
--------------

Once every partition is migrated and checked, the two collections hold the
same data and the readers are still on the source. The way you finish
depends on where the collection lives.

On a file system that renames a directory, swapping the two paths is two
renames, and nothing that reads the collection has to be told:

.. code-block:: text

    mv /data/mycollection /data/mycollection.v2
    mv /data/mycollection.v3 /data/mycollection

On an object store there is no rename: moving the copy onto the old path
means writing every key a second time. Pointing the readers at the new path
costs nothing instead, and is the usual choice there.

Either way, the files you keep at the root are yours to move: the migration does
not carry them, and a rename carries them off with the old collection.
