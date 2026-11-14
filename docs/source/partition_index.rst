.. _partition-index:

Maintaining the partition index
===============================

The partition index records the partitions known to a collection or a
view. It is updated automatically when ZCollection adds or removes
partitions.

The ``partitions()`` method lists these partitions, which can then be
selected for reading or updating.


Partitions of a view
--------------------

A view can contain only a subset of the partitions in its reference
collection. Its partition index records which partitions are included.

Calling ``sync(filters=...)`` adds the selected partitions to the view
without removing those already present.

By default, :py:meth:`~zcollection.view.View.is_synced` checks all
partitions of the reference collection. For a view covering only a
subset, specify which partitions to check using ``filters``:

.. code-block:: python

    view.is_synced(filters="year == 2026", distributed=False)

The method returns ``False`` if a selected partition is missing from
the view or its reference axis has changed. This also applies to new
partitions in the reference collection that match the filter.


Rebuilding the partition index
------------------------------

If the index no longer matches the partitions present in storage, use
``reindex()`` to rebuild it:

.. code-block:: python

    import zcollection

    collection = zcollection.open_collection("/data/measurements", mode="w")
    paths = collection.reindex()

The rebuilt index reflects the partitions currently present in storage.
The method returns their paths relative to the collection.

Reindexing only updates the partition list; it does not restore missing
data.

For example, if the index references partitions that no longer exist,
reads skip them with a warning. Rebuilding the index removes these
outdated entries.

Reindexing requires write access. If the collection uses an older
storage format, follow the :doc:`migration` procedure before opening
it for writing.

When working with views, the order of recovery matters:

- If the reference collection's index contains missing partitions,
  rebuild it before creating or synchronizing a view, or adding a
  variable to one.
- If both the reference collection and the view need reindexing,
  rebuild the reference collection's index first, then the view's.


Validating partitions
---------------------

``validate_partitions()`` checks whether the stored partitions can be
read correctly.

The method reads every array and verifies that each partition contains
an array for every variable declared in the collection's schema that
spans the partitioned dimension.

It returns the paths of invalid partitions, relative to the collection:

.. code-block:: python

    invalid = collection.validate_partitions(distributed=False)

With ``fix=True``, a collection deletes invalid partition directories
that still exist in storage and removes their entries from the index.

On a view, ``fix=True`` rewrites invalid partitions that are still present
in the reference collection and removes the others.

.. important::

   A failed ``add_variable`` can leave a variable declared in the schema
   even though some partitions lack its array.

   In this situation, call ``drop_variable`` before
   ``validate_partitions(fix=True)``. Otherwise, validation deletes the
   affected partitions of a collection or rebuilds those of a view,
   losing the values previously computed in the view's variables.

If the partition index itself is inconsistent with storage, use
``reindex()`` to rebuild it.
