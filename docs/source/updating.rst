Updating variables
==================

:py:meth:`~zcollection.collection.Collection.update` applies a function
to each partition and writes the computed values into the collection's
variables.

:py:meth:`~zcollection.collection.Collection.update_from` instead updates
variables using values from another dataset.

Both methods are also available on views, where they update the view's
own variables.

The following examples use the same nine daily measurements as
:doc:`computing`, divided into three segments containing three, two,
and four samples.

.. jupyter-execute::

    import pathlib
    import tempfile

    import fsspec
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

    def show(name, target=None):
        """Return the values of a variable, rounded."""
        loaded = (target or collection).load(selected_variables=(name, ),
                                             delayed=False,
                                             distributed=False)
        return loaded[name].values.round(2).tolist()


Writing a computed variable
---------------------------

Before updating a variable, it must already exist in the collection.
Use :py:meth:`~zcollection.collection.Collection.add_variable` to add
it to every partition, initially filled with its fill value.
See :doc:`managing` for details.

The function passed to ``update`` receives the dataset of one partition
and returns a dictionary mapping variable names to their computed arrays.
The ``variables`` parameter specifies which variables to write.

For example, add a ``doubled`` variable and populate it with twice the
values of ``signal``:

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema('doubled', 'float64', dimensions=('sample', )),
        distributed=False)

    def double(ds):
        return {'doubled': ds['signal'].values * 2}

    collection.update(double, variables=('doubled', ), delayed=False,
                      distributed=False)

``update`` returns the paths of the partitions it has written.
The new variable now contains the computed values:

.. jupyter-execute::

    show('doubled')

In this example, ``delayed=False`` loads the variables into NumPy arrays
before calling the function, while ``distributed=False`` runs the
operation without a Dask cluster.

By default, ``distributed=True`` executes each partition update in a
separate Dask task, with at most ``npartitions`` tasks in progress at
once. See :doc:`failures` for details on handling errors.

As with ``map`` in :doc:`computing`, additional positional and keyword
arguments are forwarded to the function. A function declaring a
``partition_info`` keyword parameter also receives a
:py:class:`~zcollection.PartitionInfo` object.


Updating selected partitions
----------------------------

You can update only part of a collection by selecting the partitions
to process:

- ``filters`` selects partitions using their partition keys.
- ``selected_partitions`` selects partitions by their paths.
- ``selected_variables`` specifies which variables to load.

When both ``filters`` and ``selected_partitions`` are provided, only
partitions matching both selections are updated.

For example, update ``doubled`` with three times the value of
``signal``, but only in segment 2:

.. jupyter-execute::

    def triple(ds):
        return {'doubled': ds['signal'].values * 3}

    collection.update(triple, variables=('doubled', ),
                      selected_variables=('signal', ),
                      selected_partitions=['segment=2'], delayed=False,
                      distributed=False)
    show('doubled')

The other partitions retain their previous values.


Identifying the output variables
--------------------------------

When ``variables`` is omitted, ``update`` determines which variables
to write by calling the function once on the first selected partition.

This initial call is used only to identify the variable names. Its
returned arrays are discarded, and the function is called again when
the partitions are updated.

Specify ``variables`` explicitly to avoid this extra call, especially
if the function is expensive or has side effects.

The following example lets ``update`` discover the output variable
automatically:

.. jupyter-execute::

    collection.update(double, delayed=False, distributed=False)

See :doc:`failures` for details on errors that occur during the
initial call.


Using neighboring partitions
----------------------------

Some computations require data from neighboring partitions, such as
a moving average across partition boundaries.

The ``depth`` parameter enables this behavior, following the same
principle as ``map_overlap`` in :doc:`computing`.

Each function call receives a window containing the partition being
processed and up to ``depth`` neighboring partitions on either side.
Neighbors are loaded even if they are excluded by ``filters`` or
``selected_partitions``.

Although the function receives the entire window, only the partition
being processed is updated.

By default, ``trim=True`` expects the function to return values for
the entire window. ``update`` then extracts the values corresponding
to the processed partition before writing them.

For example, calculate a three-point moving average of ``signal``
across partition boundaries and store it in a new ``smoothed``
variable:

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema('smoothed', 'float64', dimensions=('sample', )),
        distributed=False)

    kernel = numpy.full(3, 1 / 3)

    def smooth(ds):
        return {
            'smoothed': numpy.convolve(ds['signal'].values, kernel,
                                       mode='same')
        }

    collection.update(smooth, variables=('smoothed', ),
                      selected_variables=('signal', ), depth=1,
                      delayed=False, distributed=False)
    show('smoothed')

Alternatively, set ``trim=False`` when the function already returns
values for the processed partition only. Use
``partition_info.indices`` to select those values from the window.

.. note::

   With Dask, partition updates can run in any order. If a function
   reads a variable that it also writes, it may read values from
   neighboring partitions that another task has already updated.


Writing values from a dataset
-----------------------------

:py:meth:`~zcollection.collection.Collection.update_from` updates
existing variables using values from another dataset, without requiring
a computation function.

The ``align_on`` parameter identifies how rows in the input dataset
correspond to rows already stored in the collection. Each input row
updates the stored row with the same value of ``align_on``.

Rows and variables not included in the update remain unchanged.

The input dataset must contain the partitioning variables, the
``align_on`` variable, and the variables to update.

For example, replace the ``doubled`` values for two measurements
in segment 2, matching rows by their ``time`` values:

.. jupyter-execute::

    corrections = zcollection.Dataset([
        zcollection.Array('time', data['time'].values[3:5], ('sample', )),
        zcollection.Array('segment', data['segment'].values[3:5],
                          ('sample', )),
        zcollection.Array('doubled', numpy.array([-1.0, -2.0]),
                          ('sample', )),
    ], delayed=False)
    collection.update_from(corrections, align_on='time', distributed=False)

.. jupyter-execute::

    show('doubled')

Each input row must match an existing row. If a value of ``align_on``
has no match, or appears more than once among the stored or input rows
of a partition, that partition fails.

See :doc:`failures` for details on how these errors are reported.


Updating a view
---------------

Views support ``update`` and ``update_from`` for their own variables.
A view stores its data in a separate directory and references a
collection in read-only mode.

A function passed to ``view.update`` can access variables from both
the view and its source collection, but the computed values are written
only to the view.

For example, create a view with a ``centered`` variable containing
the difference between each measurement and the mean of its partition:

.. jupyter-execute::

    fs = fsspec.filesystem('file')
    view = zcollection.create_view(
        str(root / 'view'),
        zcollection.ViewReference(str(root / 'collection'), fs),
        filesystem=fs, distributed=False)
    view.add_variable(
        meta.VariableSchema('centered', 'float64', dimensions=('sample', )),
        distributed=False)

    def center(ds):
        signal = ds['signal'].values
        return {'centered': signal - signal.mean()}

    view.update(center, variables=('centered', ), delayed=False,
                distributed=False)
    show('centered', view)
