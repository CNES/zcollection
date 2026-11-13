Handling failures
=================

:py:meth:`~zcollection.collection.Collection.insert`,
:py:meth:`~zcollection.collection.Collection.update`,
:py:meth:`~zcollection.collection.Collection.update_from`,
:py:meth:`~zcollection.collection.Collection.add_variable`,
:py:meth:`~zcollection.collection.Collection.drop_variable`,
:py:meth:`~zcollection.collection.Collection.drop_partitions`,
:py:meth:`~zcollection.collection.Collection.validate_partitions` and
:py:meth:`~zcollection.collection.Collection.copy` process each
partition on its own. When some partitions fail, the operation raises
:py:class:`~zcollection.PartitionErrors`, which tells which partitions were
completed, which failed and why, and which were not attempted. Views behave the
same way for ``update``, ``update_from``, ``add_variable``, ``drop_variable``,
``sync``, ``is_synced`` and ``validate_partitions``, and so does
:py:func:`~zcollection.create_view` when it writes the partitions of a new
view. ``add_variable`` and ``drop_variable`` add and remove an immutable
variable in the calling process; ``PartitionErrors`` names it ``.immutable``.
After a failed ``is_synced``, ``completed`` names the partitions found out of
sync. After a failed check of ``validate_partitions``, ``completed`` names the
invalid partitions found, and none of them is repaired.

The examples use the same daily measurements as :doc:`updating`, with a
function that fails on segment 2.

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

Reading the report
------------------

``stop_on_error=False`` makes ``update`` attempt every partition. Segment 2
fails, the two others are written, and ``update`` raises ``PartitionErrors``.
Its attributes hold paths relative to the collection.

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), delayed=False,
                          distributed=False, stop_on_error=False)
    except zcollection.PartitionErrors as err:
        report = err

    print('completed:', report.completed)
    print('failed:', list(report.failed))
    print('not attempted:', report.not_attempted)

``failed`` maps the path of each failed partition to the traceback of its
error, as text. Its last line gives the type and the message of the error:

.. jupyter-execute::

    report.failed['segment=2'].splitlines()[-1]

The message of ``PartitionErrors`` names up to ten failed partitions, counts
the others and gives the traceback of the first, in the order the partitions
were selected. The exception objects do not reach the caller.

``update`` and ``copy`` skip a partition the index lists but the store does
not hold, with a warning; ``drop_partitions`` skips it without one. That
partition appears in none of ``completed``, ``failed`` and
``not_attempted``.

Stopping on the first failure
-----------------------------

With ``stop_on_error=True``, the default, no partition is submitted after the
first failure. With ``distributed=False``, the partitions are processed in
order, so segment 3 is not attempted:

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), delayed=False,
                          distributed=False)
    except zcollection.PartitionErrors as err:
        report = err

    print('completed:', report.completed)
    print('failed:', list(report.failed))
    print('not attempted:', report.not_attempted)

With ``distributed=True``, up to ``npartitions`` partitions are submitted at
the same time. The partitions submitted before the first failure finish, so
some partitions after the failed one may already be written.

Retrying the failed partitions
------------------------------

Once the cause is fixed, pass the paths of the report to
``selected_partitions`` to process the failed partitions again and the not
attempted ones for the first time:

.. jupyter-execute::

    def double(ds):
        return {'doubled': ds['signal'].values * 2}

    collection.update(double, variables=('doubled', ), delayed=False,
                      distributed=False,
                      selected_partitions=[*report.failed,
                                           *report.not_attempted])

A partition that ``update`` or ``update_from`` failed to write can be partly
written: an error while its variables are written leaves the ones already
written with their new values. Updating it again writes all its variables
again. If the function reads a variable it also writes, the second run reads
the values the failed run wrote, so the function must give the same result
when it runs twice.

To retry an insertion, insert the dataset again, or only its rows that fall
in the failed and not attempted partitions. Inserting the whole dataset again
writes it again into every partition, including those already written; with
a merge callable, the result is the same only if the callable gives the same
result when it runs twice.

A failed ``add_variable`` removes what it added, unless that removal fails
too. The ``completed`` attribute of the error names the partitions that
received the array before the removal. A failed ``drop_variable`` is
finished by calling it again. See :doc:`managing`.

To retry ``drop_partitions``, call it again with the same ``filters`` and
``timedelta``: the partitions already removed are no longer listed, and the
others are removed. ``timedelta`` is measured from the time of the call, so
the retry can select more partitions.

To retry ``copy``, remove its target, or copy to another location.

To retry ``View.sync``, call it again with the same ``filters``: the
partitions already synchronized are skipped. After an interruption, call
``validate_partitions(fix=True)`` first, to write again a partition left
partly written.

``View.is_synced`` raises ``PartitionErrors`` for a partition it cannot
check. ``validate_partitions(fix=True)`` writes again a partition without
its ``.checksum`` array, which holds the reference axis the partition was
written from.

A failed ``create_view`` leaves the view with the partitions it completed.
Open it with :py:func:`~zcollection.open_view`, call
``validate_partitions(fix=True)`` to write again a partition left partly
written, then ``sync(filters=view.filters)``, because ``sync()`` without
``filters`` synchronizes every partition of the reference collection.

Errors raised before any partition starts
-----------------------------------------

An error found before the operation processes any partition is raised
directly, without ``PartitionErrors``: a dataset that does not match the
collection, a value of ``npartitions`` below one, a partition that
``update_from`` cannot find, or a ``copy`` to a target that is not empty
raises ``ValueError``. An error raised by the
function of ``update`` while it infers its output variables arrives as a
``RuntimeError``.

.. jupyter-execute::

    try:
        collection.update(double, variables=('doubled', ), npartitions=0,
                          distributed=False)
    except ValueError as err:
        print(err)

What the operations guarantee
-----------------------------

**The lock.** When a collection or a view has a synchronizer, such as
:py:class:`~zcollection.sync.ProcessSync`, the operation takes the lock while
it writes partitions, and releases it once they have finished, including
after a failure. With ``distributed=True``, an interrupted operation, for
example with :kbd:`Ctrl-C`, waits for the tasks it submitted before it
releases the lock. A second interruption during that wait releases the lock
at once, while those tasks may still be writing. With ``distributed=False``,
an interruption stops the partition in progress. ``copy`` takes only the
lock of its ``synchronizer`` argument, which the new collection receives.

**The partition index.** When ``insert``, ``drop_partitions``, ``copy``,
``View.sync`` or ``create_view`` stops on an error or an interruption, the
partitions listed afterwards are the ones the store holds, including one
left partly written or partly removed. For ``copy``, these are the
partitions of the new collection.

**Inserted partitions.** ``insert`` writes a new partition in place and
removes it when its writing fails. To replace an existing partition, it
writes the merged partition to a temporary directory, removes the stored one
and moves the new one in its place. An error before the move leaves the
stored partition unchanged; an error during the move can leave it incomplete
or absent.

**Lost workers.** If a Dask worker is lost, Dask runs its task again on
another worker, up to the number of times
``distributed.scheduler.allowed-failures`` sets. A partition can then be
processed more than once. It is listed as failed when its task has lost its
worker more times than that. A worker lost while writing can leave the
partition in an unknown state;
:py:meth:`~zcollection.collection.Collection.validate_partitions` lists the
partitions that cannot be read.
