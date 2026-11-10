.. _writing-data:

Writing data
============

:py:meth:`~zcollection.collection.Collection.insert`,
:py:meth:`~zcollection.collection.Collection.update` and
:py:meth:`~zcollection.collection.Collection.update_from` write partitions.
Views write theirs with :py:meth:`~zcollection.view.View.update` and
:py:meth:`~zcollection.view.View.update_from`. The three operations return
the paths of the partitions they wrote, relative to the collection or the
view. When a partition fails, they raise
:py:class:`~zcollection.PartitionErrors`, which lists the partitions written,
failed and not attempted.

Parallelism
-----------

With ``distributed=True``, each partition is processed in a Dask task of its
own. The operation submits ``npartitions`` tasks, then one more each time a
task finishes. ``npartitions`` defaults to the number of Dask workers.

With ``distributed=True``, ``insert`` also writes the variables of a partition
in parallel, on the worker that writes that partition. ``npartitions`` limits
the partitions written at the same time; each partition in progress can use
every thread of its worker for its variables.

With ``distributed=False``, the partitions are processed one after the other
in the calling process, and ``npartitions`` has no effect. In both modes, the
operation raises ``ValueError`` when ``npartitions`` is below one.

Failures
--------

If an error is raised while a partition is processed (by the callback of
``update``, the merge callable of ``insert``, the matching of rows in
``update_from``, or the reading or writing of the partition), the operation
raises :py:class:`zcollection.PartitionErrors` once every partition it
started has finished.
Its attributes hold paths relative to the collection or the view:

``written``
    The partitions written.
``failed``
    The traceback of the error of each failed partition, keyed by path.
``not_attempted``
    The partitions the operation did not attempt.

.. code-block:: python

    try:
        collection.update(compute_flags, variables=('flags', ))
    except zcollection.PartitionErrors as err:
        print('written:', err.written)
        for partition, trace in err.failed.items():
            print('failed:', partition)
            print(trace)
        print('not attempted:', err.not_attempted)

The message of ``PartitionErrors`` names up to ten failed partitions and
gives the traceback of the first one, in the order the operation processes the
partitions. ``failed`` holds each traceback as text; the exception objects do
not reach the caller.

Errors found before any partition starts are raised directly: a dataset that
does not match the collection, or a partition that ``update_from`` cannot
find, raises ``ValueError``. When ``variables`` is omitted, ``update`` calls
the callback once to infer its output variables; an error raised by the
callback during that call arrives as a ``RuntimeError``.

``update`` skips a partition the index lists but the store does not hold,
with a warning. That partition is not listed as written or failed.

Stopping on error
-----------------

With ``stop_on_error=True``, the default, no partition starts after the first
failure, and the partitions in progress finish. The partitions left over
are listed in ``not_attempted``. With ``stop_on_error=False``, the operation
starts every partition, and ``not_attempted`` is empty.

With ``distributed=True``, some of the partitions after the failed one may
already be written.

Retrying failed partitions
--------------------------

A partition that ``update`` or ``update_from`` failed to write can be partly
written: an error while its variables are written leaves the ones already
written with their new values. Updating it again writes all its variables
again. If the callback reads a variable it also writes, the second run reads
the values the failed run wrote, so the callback must give the same result
when it runs twice.

Select the partitions to update again with ``selected_partitions``, which
takes the paths ``PartitionErrors`` lists:

.. code-block:: python

    try:
        collection.update(compute_flags, variables=('flags', ))
    except zcollection.PartitionErrors as err:
        collection.update(compute_flags, variables=('flags', ),
                          selected_partitions=[*err.failed,
                                               *err.not_attempted])

With ``depth``, the window of a partition contains only selected partitions.
To give the failed partitions their full windows, select their neighbors too;
``update`` then writes the neighbors again.

``insert`` writes a new partition in place and removes it when its writing
fails. To replace an existing partition, it writes the merged partition to a
temporary directory, removes the stored one and moves the new one in its
place. An error before the move leaves the stored partition unchanged; an
error during the move can leave it incomplete or absent. Inserting the dataset
again merges it again into every partition, including those already written,
so with a merge callable it gives the same result only if the callable does
when it runs twice.

Locking
-------

When a collection or a view has a synchronizer, such as
:py:class:`~zcollection.sync.ProcessSync`, the operation holds its lock while
it writes partitions, and releases it once they have finished, including after
a failure.

With ``distributed=True``, an interrupted operation, for example with
:kbd:`Ctrl-C`, waits for the tasks it submitted before it releases the lock.
A second interruption during that wait releases the lock at once, while those
tasks may still be writing. With ``distributed=False``, an interruption stops
the partition in progress.

Lost workers
------------

If a Dask worker is lost, Dask runs its task again on another worker, up to
the number of times ``distributed.scheduler.allowed-failures`` sets. A
partition can then be processed more than once. It is listed as failed when its task
has lost its worker more times than that. A worker lost while writing can
leave the partition in an unknown state;
:py:meth:`~zcollection.collection.Collection.validate_partitions` lists the
partitions that cannot be read.

Merge arguments
---------------

The keyword arguments of ``insert`` that are not parameters of ``insert``
reach the merge callable, such as the ``tolerance`` of
:py:func:`~zcollection.merging.time_series.merge_time_series`:

.. code-block:: python

    collection.insert(ds,
                      merge_callable=zcollection.merging.merge_time_series,
                      tolerance=numpy.timedelta64(1, 'h'))

An argument named after a positional parameter of the merge callable, such
as ``axis``, makes each partition that already holds data fail with
``TypeError``. New partitions are written without the merge callable.
