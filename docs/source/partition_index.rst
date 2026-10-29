.. _partition-index:

The partition index
===================

The partition index is the list of partitions known to a collection or a
view. ``partitions()`` returns this list, and you can select partitions
from it for reading or updating. When zcollection adds or removes
partitions, it updates the index.

The partitions of a view
------------------------

A view can cover a subset of its reference collection. Its partition
index records that selection. Calling ``sync(filters=...)`` adds the
selected partitions to the view and retains those already included.

:py:meth:`~zcollection.view.View.is_synced` checks all reference partitions
by default. For a partial view, pass the filter selecting the partitions
to check:

.. code-block:: python

    view.is_synced(filters="year == 2026", distributed=False)

The check returns ``False`` if a selected partition is absent from the
view or its reference axis has changed. This includes new reference
partitions matching the filter.

Rebuilding the index
---------------------

If the index no longer matches the partitions present in the store,
use ``reindex()`` to reconstruct it:

.. code-block:: python

    import zcollection

    collection = zcollection.open_collection("/data/measurements", mode="w")
    paths = collection.reindex()

The rebuilt index lists the partitions present in the store. ``paths``
contains their relative paths. Rebuilding the index does not restore
missing data.

If the index lists partitions that are missing from the store, reads skip
them and log a warning. Rebuilding the index removes those entries.

Reindexing requires write access. For a collection in an older storage
format, follow the :doc:`migration` procedure before opening it for writing.

If the reference index lists missing partitions, rebuild it before
creating or synchronizing a view, or adding a variable to one. When both
indexes need repair, rebuild the reference index first, then the view's.

Checking partitions
--------------------

To find partitions that cannot be opened or read, use
``validate_partitions()``. It reads every array and returns the paths of
the invalid partitions:

.. code-block:: python

    invalid = collection.validate_partitions(distributed=False)

With ``fix=True``, it deletes invalid partition directories still present
in the store and removes their index entries. Use ``reindex()`` to rebuild
the partition list when the index needs repair.
