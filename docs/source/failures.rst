Handling failures
=================

ZCollection processes partitions independently for operations such as
:py:meth:`~zcollection.collection.Collection.insert`,
:py:meth:`~zcollection.collection.Collection.update`,
:py:meth:`~zcollection.collection.Collection.update_from`,
:py:meth:`~zcollection.collection.Collection.add_variable`,
:py:meth:`~zcollection.collection.Collection.drop_variable`,
:py:meth:`~zcollection.collection.Collection.drop_partitions`,
:py:meth:`~zcollection.collection.Collection.validate_partitions`, and
:py:meth:`~zcollection.collection.Collection.copy`.

If one or more partitions fail, the operation raises
:py:class:`~zcollection.PartitionErrors`. This exception reports which
partitions completed, which failed and why, and which were not attempted.

The same behavior applies to the ``update``, ``update_from``,
``add_variable``, ``drop_variable``, ``sync``, ``is_synced``, and
``validate_partitions`` methods of views, as well as
:py:func:`~zcollection.create_view` when creating a view's partitions.

``add_variable`` and ``drop_variable`` handle immutable variables in the
calling process. In ``PartitionErrors``, these variables are identified
by ``.immutable``.

The following examples use the same daily measurements as :doc:`updating`,
with a function that deliberately fails on segment 2.

.. jupyter-execute::

    import pathlib
    import tempfile

    import numpy
    import zcollection
    from zcollection import meta

    root = pathlib.Path(tempfile.mkdtemp())
    origin = numpy.datetime64('2026-01-01', 'D')
    data = zcollection.Dataset([
        zcollection.Array(
            'time', origin + numpy.arange(9).astype('timedelta64[D]'),
            ('sample', )),
        zcollection.Array(
            'segment', numpy.repeat([1, 2, 3], [3, 2, 4]), ('sample', )),
        zcollection.Array(
            'signal', numpy.arange(9, dtype='float64') ** 2, ('sample', )),
    ], delayed=False)
    collection = zcollection.create_collection(
        'time', data, zcollection.partitioning.Sequence(('segment', )),
        str(root / 'collection'))
    collection.insert(data, distributed=False)
    collection.add_variable(
        meta.VariableSchema('doubled', 'float64', dimensions=('sample', )),
        distributed=False)

    def double(ds):
        if ds['segment'].values[0] == 2:
            raise ValueError('segment 2 cannot be doubled')
        return {'doubled': ds['signal'].values * 2}


Reading the failure report
--------------------------

By setting ``stop_on_error=False``, an operation attempts every partition,
even if some fail.

In the following example, segment 2 raises an error while segments 1
and 3 are updated successfully. The operation then raises
``PartitionErrors``.

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), delayed=False,
                          distributed=False, stop_on_error=False)
    except zcollection.PartitionErrors as err:
        report = err

    print('completed:', report.completed)
    print('failed:', list(report.failed))
    print('not attempted:', report.not_attempted)

The exception provides three attributes containing partition paths
relative to the collection:

- ``completed`` lists the partitions that completed successfully.
- ``failed`` maps each failed partition to its error traceback, as text.
- ``not_attempted`` lists the partitions that were not processed.

The last line of each traceback contains the error type and message:

.. jupyter-execute::

    report.failed['segment=2'].splitlines()[-1]

The ``PartitionErrors`` message includes the names of up to ten failed
partitions, the number of additional failures, and the traceback of the
first failure in partition selection order.

The original exception objects are not returned to the caller.

Some operations give ``completed`` a specific meaning:

- After a failed ``View.is_synced``, it lists partitions found out of sync.
- After a failed ``validate_partitions`` check, it lists invalid
  partitions found during the check. None of these partitions is repaired.

Missing partitions are handled separately. If a partition is listed in
the index but absent from storage, ``update`` and ``copy`` skip it with
a warning, while ``drop_partitions`` skips it silently.

Such partitions appear in none of the three report attributes.


Stopping on the first failure
-----------------------------

By default, ``stop_on_error=True`` prevents new partitions from being
submitted after the first failure.

With ``distributed=False``, partitions are processed sequentially.
In this example, segment 2 fails, so segment 3 is never attempted:

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), delayed=False,
                          distributed=False)
    except zcollection.PartitionErrors as err:
        report = err

    print('completed:', report.completed)
    print('failed:', list(report.failed))
    print('not attempted:', report.not_attempted)

With ``distributed=True``, up to ``npartitions`` tasks can be in
progress at once. Tasks already submitted when the first failure
occurs are allowed to finish.

Consequently, some partitions following the failed one may already
have been written, even with ``stop_on_error=True``.


Retrying failed operations
--------------------------

Once the cause of a failure has been resolved, the recovery procedure
depends on the operation.

Updating variables
~~~~~~~~~~~~~~~~~~

For ``update``, use the paths in ``failed`` and ``not_attempted`` to
retry failed partitions and process those that were skipped.

For example, after fixing the function that failed on segment 2:

.. jupyter-execute::

    def double(ds):
        return {'doubled': ds['signal'].values * 2}

    collection.update(double, variables=('doubled', ), delayed=False,
                      distributed=False,
                      selected_partitions=[*report.failed,
                                           *report.not_attempted])

A failed ``update`` or ``update_from`` can leave a partition partially
updated. If an error occurs while writing its variables, those written
before the failure retain their new values.

Retrying the operation writes all the variables again.

If the function reads a variable that it also writes, the retry may
use values already modified by the failed attempt. Such functions must
therefore produce the same result when executed more than once.

Inserting data
~~~~~~~~~~~~~~

To retry ``insert``, insert the original dataset again, or select only
the rows belonging to failed and not attempted partitions.

Reinserting the entire dataset also rewrites partitions that completed
successfully.

When using a merge callable, this produces the same result only if
applying the callable more than once does not change the result.

Managing variables and partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A failed ``add_variable`` attempts to remove the arrays it created.
If this cleanup also fails, the variable remains declared in the schema.

The ``completed`` attribute identifies the partitions that received
the array before the cleanup attempt. See :doc:`managing` for details
on recovery.

A failed ``drop_variable`` can be completed by calling it again.

To retry ``drop_partitions``, call it again with the same ``filters``
and ``timedelta`` parameters. Partitions already removed are no longer
listed, so the operation processes those that remain.

Keep in mind that ``timedelta`` is evaluated relative to the time of
each call. A retry may therefore select additional partitions.

To retry ``copy``, remove the target collection or choose another
destination.

Recovering views
~~~~~~~~~~~~~~~~

To retry ``View.sync``, call it again with the same ``filters``.
Partitions already synchronized are skipped.

After an interruption, first call ``validate_partitions(fix=True)``
to rewrite any partition left partially written.

The ``.checksum`` array records the reference axis from which a
partition was written. If this array is missing,
``validate_partitions(fix=True)`` rewrites the partition.

``View.is_synced`` raises ``PartitionErrors`` when it cannot check
a partition.

If ``create_view`` fails, the new view remains with the partitions
that were successfully created.

To recover it:

1. Open the view using :py:func:`~zcollection.open_view`.
2. Call ``validate_partitions(fix=True)`` to repair partitions left
   partially written.
3. Call ``sync(filters=view.filters)`` to synchronize the remaining
   partitions.

The ``filters`` argument is important here: calling ``sync()``
without it synchronizes every partition of the reference collection.


Errors raised before processing partitions
------------------------------------------

Not every error is reported through ``PartitionErrors``.

Errors detected before partition processing begins are raised directly.
For example, ``ValueError`` is raised when:

- An input dataset does not match the collection.
- ``npartitions`` is less than 1.
- ``update_from`` cannot find a required partition.
- ``copy`` targets a non-empty location.

If the function passed to ``update`` fails while its output variables
are being inferred, the error is reported as a ``RuntimeError``.

For example, an invalid ``npartitions`` value raises ``ValueError``
directly:

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), npartitions=0,
                          distributed=False)
    except ValueError as err:
        print(err)


What happens after a failure
----------------------------

Understanding how operations modify storage is important when
recovering from failures or interruptions.

Locks
~~~~~

When a collection or view has a synchronizer, such as
:py:class:`~zcollection.sync.ProcessSync`, write operations hold its
lock while processing partitions.

The lock is released once the submitted work has finished, including
after a failure.

With ``distributed=True``, an interruption such as :kbd:`Ctrl-C`
waits for already submitted tasks to finish before releasing the lock.

A second interruption during that wait releases the lock immediately,
even if some tasks are still writing.

With ``distributed=False``, an interruption stops the partition
currently being processed.

The ``copy`` operation only acquires the lock provided through its
``synchronizer`` argument. This synchronizer is also assigned to the
new collection.

Partition index
~~~~~~~~~~~~~~~

If ``insert``, ``drop_partitions``, ``copy``, ``View.sync``, or
``create_view`` stops because of an error or interruption, the
partitions listed afterwards are those present in storage.

This includes partitions that may have been left partially written
or partially removed.

For ``copy``, this refers to the partitions of the new collection.

Inserting and replacing partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When inserting a new partition, ``insert`` writes it directly to its
destination. If writing fails, the new partition is removed.

Replacing an existing partition follows a different process:

1. The new or merged partition is written to a temporary directory.
2. The existing partition is removed.
3. The temporary partition is moved into its place.

An error before the move leaves the existing partition unchanged.
An error during the move can leave the partition incomplete or absent.

Lost Dask workers
~~~~~~~~~~~~~~~~~

If a Dask worker is lost, Dask can retry its task on another worker,
up to the limit set by ``distributed.scheduler.allowed-failures``.

A partition may therefore be processed more than once. Its task is
reported as failed when worker losses exceed the configured limit.

If a worker is lost while writing, the partition may be left in an
unknown state.

Use :py:meth:`~zcollection.collection.Collection.validate_partitions`
to identify partitions that can no longer be read.
