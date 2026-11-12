Managing variables and partitions
=================================

:py:meth:`~zcollection.collection.Collection.add_variable` and
:py:meth:`~zcollection.collection.Collection.drop_variable` add and remove
variables, and
:py:meth:`~zcollection.collection.Collection.drop_partitions` removes
partitions from a collection.

The examples use daily measurements split into segments of three, two and
four samples.

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

``add_variable`` declares a variable in the schema of the collection and
creates its array, filled with its fill value. Where the array is created
depends on its dimensions:

- A variable with the partitioned dimension gets an array in every
  partition.
- A variable without it, an immutable variable, gets a single array, in
  ``.immutable``.

For example, to add a quality flag to every sample:

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema('quality', 'int8', dimensions=('sample', ),
                            fill_value=-1),
        distributed=False)
    loaded = collection.load(selected_variables=('quality', ), delayed=False,
                             distributed=False)
    loaded['quality'].values

With a ``fill_value``, every value is masked until the variable is written,
for example with :py:meth:`~zcollection.collection.Collection.update`; see
:doc:`updating`. Without one, every value is the default value of its data
type in Zarr: zero for a number, ``False`` for a boolean, ``NaT`` for a date
or a duration, and an empty string for a string.

The collection must not already have a variable of that name, and the
variable must use only dimensions the collection declares; otherwise
``add_variable`` raises ``ValueError``.

Removing a variable
-------------------

``drop_variable`` removes the array of a variable, then the variable from the
schema:

.. jupyter-execute::

    collection.drop_variable('quality', distributed=False)
    list(collection.metadata.variables)

A variable used by the partitioning cannot be removed.

Views
-----

A view provides ``add_variable`` and ``drop_variable`` for its own
variables. It refuses a variable without the partitioned dimension, and
cannot remove a variable of the collection it references.

When adding or removing a variable fails
----------------------------------------

If a partition fails, ``add_variable`` removes the arrays it created and
the variable from the schema, then raises
:py:class:`~zcollection.PartitionErrors`. If that removal fails as well, the
schema still declares the variable, and the error carries a note naming the
partitions where its array may remain. Call ``drop_variable`` before reading
the collection or the view: until then, some partitions lack an array the
schema declares.

If a partition fails, ``drop_variable`` raises ``PartitionErrors`` and the
schema still declares the variable. Calling ``drop_variable`` again removes
the arrays that remain.

Removing partitions
-------------------

Partitions can be selected using ``filters``, ``timedelta``, or both:

- ``filters`` selects partitions based on their partition keys.
- ``timedelta`` selects partitions older than the specified duration, based
  on timestamps reported by the file system.

When both parameters are provided, only partitions matching both conditions
are removed. If neither is specified, all partitions are removed, while the
configuration of the collection and its immutable variables are preserved.

For example, to remove the partition where ``segment == 2``:

.. jupyter-execute::

    collection.drop_partitions(filters='segment == 2', distributed=False)

.. jupyter-execute::

    list(collection.partitions(relative=True))

The method returns the paths of the removed partitions, relative to the
collection. Partitions listed in the index but missing from the store are
excluded from the returned paths. See :doc:`failures` for how to retry a
failed ``drop_partitions``.

Timestamps of the file system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **S3:** creation timestamps are not available, so using ``timedelta``
  raises ``NotImplementedError``.
- **Local file systems on Linux:** the reported timestamp reflects the last
  change to the partition directory rather than its actual creation time.
  Inserting data into a partition, for example, resets its age.

Distributed execution
---------------------

With ``distributed=True``, the default, ``add_variable``, ``drop_variable``
and ``drop_partitions`` process each partition in a Dask task of its own,
and at most ``npartitions`` tasks run at the same time. An immutable
variable is added and removed in the calling process.
