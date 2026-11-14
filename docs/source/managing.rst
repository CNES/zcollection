Managing variables and partitions
=================================

ZCollection provides methods to manage variables and partitions in an
existing collection:

- :py:meth:`~zcollection.collection.Collection.add_variable` adds a variable.
- :py:meth:`~zcollection.collection.Collection.drop_variable` removes a variable.
- :py:meth:`~zcollection.collection.Collection.drop_partitions` removes partitions.

The following examples use nine daily measurements divided into three
segments containing three, two, and four samples.

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
    list(collection.partitions(relative=True))


Adding a variable
-----------------

:py:meth:`~zcollection.collection.Collection.add_variable` adds a variable
to the collection's schema and creates its array, initialized with its
fill value.

Where the array is stored depends on the variable's dimensions:

- Variables spanning the partitioned dimension are stored separately
  in each partition.
- Variables without the partitioned dimension are immutable and stored
  only once, in the ``.immutable`` directory.

For example, add a ``quality`` flag to every sample, initially set
to ``-1``:

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema('quality', 'int8', dimensions=('sample', ),
                            fill_value=-1),
        distributed=False)
    loaded = collection.load(selected_variables=('quality', ), delayed=False,
                             distributed=False)
    loaded['quality'].values

When a ``fill_value`` is specified, all values are initially masked
until data is written to the variable, for example using
:py:meth:`~zcollection.collection.Collection.update`. See :doc:`updating`
for details.

Without a ``fill_value``, Zarr initializes the array with the default
value for its data type: zero for numbers, ``False`` for booleans,
``NaT`` for dates and durations, and an empty string for strings.

The new variable must have a unique name and use only dimensions
declared by the collection. Otherwise, ``add_variable`` raises
``ValueError``.


Removing a variable
-------------------

:py:meth:`~zcollection.collection.Collection.drop_variable` removes
a variable's stored arrays and its declaration from the collection's
schema.

For example, remove the ``quality`` variable:

.. jupyter-execute::

    collection.drop_variable('quality', distributed=False)
    list(collection.metadata.variables)

Variables used by the partitioning scheme cannot be removed.


Managing variables in views
---------------------------

Views also provide ``add_variable`` and ``drop_variable`` to manage
their own variables.

Views cannot contain immutable variables: every variable of a view
must span the partitioned dimension.

A view also cannot remove variables belonging to its source collection,
which it references in read-only mode.


Handling variable management failures
-------------------------------------

Adding or removing a variable may fail while processing individual
partitions. The recovery behavior depends on the operation.

**Adding a variable**

If a partition fails during ``add_variable``, ZCollection attempts to
remove the arrays already created and revert the schema change before
raising :py:class:`~zcollection.PartitionErrors`.

If this cleanup also fails, the variable remains declared in the schema.
The exception includes a note identifying the partitions where its
array may still exist.

In this situation, call ``drop_variable`` before reading the collection
or view. Until cleanup is complete, some partitions may be missing an
array declared by the schema.

**Removing a variable**

If a partition fails during ``drop_variable``, the method raises
:py:class:`~zcollection.PartitionErrors` and leaves the variable
declared in the schema.

Call ``drop_variable`` again to remove the remaining arrays and
complete the operation.


Removing partitions
-------------------

:py:meth:`~zcollection.collection.Collection.drop_partitions` removes
partitions from a collection. Partitions can be selected using
``filters``, ``timedelta``, or both:

- ``filters`` selects partitions based on their partition keys.
- ``timedelta`` selects partitions older than the specified duration,
  based on timestamps reported by the file system.

When both parameters are provided, only partitions matching both
conditions are removed. If neither is specified, all partitions are
removed, while the collection's configuration and immutable variables
are preserved.

For example, remove the partition where ``segment == 2``:

.. jupyter-execute::

    collection.drop_partitions(filters='segment == 2', distributed=False)

The remaining partitions can be listed with:

.. jupyter-execute::

    list(collection.partitions(relative=True))

The method returns the paths of the removed partitions, relative to the
collection. Partitions listed in the index but missing from the store
are excluded from the returned paths.

See :doc:`failures` for details on how to retry a failed
``drop_partitions`` operation.


File system timestamps
^^^^^^^^^^^^^^^^^^^^^^

The ``timedelta`` parameter relies on timestamps provided by the
underlying file system.

- **S3:** Creation timestamps are not available, so using ``timedelta``
  raises ``NotImplementedError``.
- **Local file systems on Linux:** The reported timestamp reflects the
  last change to the partition directory rather than its actual
  creation time. Inserting data into a partition, for example, resets
  its age.


Distributed execution
---------------------

By default, ``add_variable``, ``drop_variable``, and
``drop_partitions`` use Dask (``distributed=True``).

Each partition is processed in a separate task, with at most
``npartitions`` tasks in progress at once.

Immutable variables are handled differently: they are added and removed
directly in the calling process, without a Dask task.
