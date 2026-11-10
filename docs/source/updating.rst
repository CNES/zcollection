Updating variables
==================

:py:meth:`~zcollection.collection.Collection.update` calls a function on each
partition and writes what it returns into variables of the collection.
:py:meth:`~zcollection.collection.Collection.update_from` writes the values a
dataset holds. Views provide both methods for their own variables.

The examples use the same daily measurements as :doc:`computing`, split into
segments of three, two and four samples.

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

The variable must exist before ``update`` writes it:
:py:meth:`~zcollection.collection.Collection.add_variable` adds it to every
partition, filled with its fill value. The function receives the dataset of
one partition and returns a mapping of variable names to arrays.
``variables`` names the variables it writes.

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema('doubled', 'float64', dimensions=('sample', )),
        distributed=False)

    def double(ds):
        return {'doubled': ds['signal'].values * 2}

    collection.update(double, variables=('doubled', ), delayed=False,
                      distributed=False)

``update`` returns the paths of the partitions it wrote. The variable now
holds the computed values:

.. jupyter-execute::

    show('doubled')

With ``delayed=False``, the variables are loaded into NumPy arrays before the
call; with ``distributed=False``, the calls run in the calling process. With
``distributed=True``, the default, each partition is updated in a Dask task
of its own, at most ``npartitions`` at the same time; see :doc:`failures` for
what happens when one of them fails.

The function also receives the arguments following it in the call, the extra
keyword arguments, and a :py:class:`~zcollection.PartitionInfo` if it declares
a ``partition_info`` parameter, as with ``map`` in :doc:`computing`.

Updating some partitions
------------------------

``filters`` selects the partitions by their keys, and
``selected_partitions`` by their paths. ``selected_variables`` lists the
variables to load, as in :doc:`computing`.

.. jupyter-execute::

    def triple(ds):
        return {'doubled': ds['signal'].values * 3}

    collection.update(triple, variables=('doubled', ),
                      selected_variables=('signal', ),
                      selected_partitions=['segment=2'], delayed=False,
                      distributed=False)
    show('doubled')

Naming the outputs
------------------

Without ``variables``, ``update`` calls the function once on the first
selected partition to find the names it returns, then discards the arrays.
Pass ``variables`` when this extra call is expensive or has side effects; an
error during that call is described in :doc:`failures`.

.. jupyter-execute::

    collection.update(double, delayed=False, distributed=False)

Using neighboring partitions
----------------------------

With ``depth``, the function receives each partition with its neighbors, as
``map_overlap`` does, and returns values for the whole window. With
``trim=True``, the default, ``update`` keeps the values of the processed
partition.

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

Use ``trim=False`` when the function returns the values of the processed
partition only, selected with ``partition_info.indices``.

The calls run in any order with Dask. A function that reads a variable it
also writes can read, in a neighbor, values another call has already written.

Writing values from a dataset
-----------------------------

``update_from`` writes the variables of a dataset into the partitions its
rows fall in. ``align_on`` names the variable identifying a row: each row of
the dataset is written to the stored row with the same value. The other rows
and variables keep their values. The dataset must contain the partitioning
variables, ``align_on`` and the variables to write.

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

A row whose value of ``align_on`` matches no stored row, or a value repeated
among the stored rows or the dataset rows of a partition, makes that
partition fail; see :doc:`failures`.

Updating a view
---------------

A view stores its own variables beside a collection it references read
only. The function given to ``view.update`` reads the variables of the view
and of the collection, and ``update`` writes the variables of the view.

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
