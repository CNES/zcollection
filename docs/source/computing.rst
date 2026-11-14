Computing on partitions
=======================

:py:meth:`~zcollection.collection.Collection.map` applies a function to
each partition and returns a Dask bag containing the results.

:py:meth:`~zcollection.collection.Collection.map_overlap` works similarly,
but also loads neighboring partitions to provide additional context for
the computation.

Both methods read data without modifying the collection. To write
computed results back to variables, see :doc:`updating`. The same methods
are also available on views.

The following examples use nine consecutive days of measurements,
divided into three segments containing three, two, and four samples.

.. jupyter-execute::

    import pathlib
    import tempfile

    import numpy
    import zcollection

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


Computing one value per partition
---------------------------------

With ``map``, the function receives the dataset of one partition.
The returned Dask bag can then be computed to obtain the results.

For example, calculate the mean of ``signal`` in each partition:

.. jupyter-execute::

    def mean(ds):
        return float(ds['signal'].values.mean())

    bag = collection.map(mean, delayed=False)
    bag.compute(scheduler='synchronous')

Each result contains the partition keys and the value returned by
the function.

Here, ``delayed=False`` loads the variables into NumPy arrays before
calling the function. See `Choosing NumPy or Dask arrays`_ for details.

The ``compute`` method determines how the bag is executed. In this
example, ``scheduler='synchronous'`` runs the computation in the calling
process. Without an explicit scheduler, Dask uses the current client's
cluster if one is available, or the default scheduler for Dask bags.


Choosing the inputs
-------------------

You can control which variables and partitions are passed to the function:

- ``selected_variables`` specifies which variables to load.
- ``filters`` selects partitions based on their partition keys.
- ``selected_partitions`` selects partitions by their paths.

If both ``filters`` and ``selected_partitions`` are provided, only
partitions matching both selections are processed.

For example, compute the mean using only ``signal`` and selecting
segments 2 and 3:

.. jupyter-execute::

    bag = collection.map(mean, selected_variables=('signal', ),
                         filters='segment >= 2', delayed=False)
    bag.compute(scheduler='synchronous')

You can also select partitions explicitly by their paths:

.. jupyter-execute::

    bag = collection.map(mean, selected_variables=('signal', ),
                         selected_partitions=['segment=1', 'segment=3'],
                         delayed=False)
    bag.compute(scheduler='synchronous')

Additional positional and keyword arguments are forwarded to the
function. For example, to multiply each mean by a given factor:

.. jupyter-execute::

    def scaled_mean(ds, scale):
        return float(ds['signal'].values.mean() * scale)

    collection.map(scaled_mean, 2.0, selected_variables=('signal', ),
                   delayed=False).compute(scheduler='synchronous')


Knowing which partition is processed
------------------------------------

A function can access information about the partition being processed
by declaring a ``partition_info`` keyword parameter.

ZCollection then passes a :py:class:`~zcollection.PartitionInfo` object
containing the partition's keys, path, dimension, and indices.

For example, retrieve the name and keys of each partition:

.. jupyter-execute::

    def describe(ds, *, partition_info):
        return partition_info.name, partition_info.partition

    collection.map(describe, selected_variables=('signal', ),
                   delayed=False).compute(scheduler='synchronous')

The ``PartitionInfo`` object provides the following attributes:

.. list-table:: Attributes of PartitionInfo
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Meaning
   * - ``partition``
     - Mapping of the names of the partitioning variables to their values,
       such as ``{'segment': 2}``.
   * - ``name``
     - Path of the partition relative to the collection, as
       ``selected_partitions`` takes it.
   * - ``dimension``
     - Dimension along which the data is partitioned, ``sample`` here.
   * - ``indices``
     - Slice selecting the partition within the data the function receives:
       ``slice(None)`` without neighbors.

See :doc:`partitioning` for details on how partition keys and names
are defined.


Using neighboring partitions
----------------------------

Some computations require data from neighboring partitions, such as
calculating a moving average across partition boundaries.

``map_overlap`` provides this context by loading each partition together
with up to ``depth`` neighboring partitions on either side. Their data
is concatenated along the partitioned dimension before being passed
to the function.

The ``filters`` and ``selected_partitions`` parameters determine which
partitions are processed. However, neighboring partitions are still
loaded when needed, even if they are not part of the selection.

The function receives the entire window, while
``partition_info.indices`` identifies the samples belonging to the
partition being processed.

For the example collection, ``depth=1`` produces these windows:

.. list-table:: Windows and central slices
   :header-rows: 1
   :widths: 16 30 24 30

   * - Segment
     - Segments in the window
     - Samples in the window
     - ``partition_info.indices``
   * - 1
     - 1, 2
     - 5
     - ``slice(0, 3)``
   * - 2
     - 1, 2, 3
     - 9
     - ``slice(3, 5)``
   * - 3
     - 2, 3
     - 6
     - ``slice(2, 6)``

The following example computes a three-point moving average. Each
partition can use samples from its neighbors, allowing the smoothing
operation to cross partition boundaries.

The function then uses ``partition_info.indices`` to return only the
values belonging to the partition being processed. At the ends of
the signal, missing samples are treated as zeros.

.. jupyter-execute::

    kernel = numpy.full(3, 1 / 3)

    def smooth(ds, *, partition_info):
        window = numpy.convolve(ds['signal'].values, kernel, mode='same')
        return window[partition_info.indices].round(2).tolist()

    collection.map_overlap(smooth, depth=1, selected_variables=('signal', ),
                           delayed=False).compute(scheduler='synchronous')

The meaning of ``partition_info.indices`` remains the same even when
``selected_variables`` excludes variables spanning the partitioned
dimension, for example when selecting only an immutable variable.


Reading an external Dask array
------------------------------

Functions passed to ``map`` can also receive external Dask arrays.

As with :py:meth:`dask.bag.Bag.map`, a Dask array passed as an argument
remains the same lazy array in each call. The function can then select
and compute only the data it needs.

For example, suppose an external array contains one weight for each
day of the collection. To calculate a weighted mean for each partition,
the function must select the weights corresponding to its measurements.

The ``partition_info.indices`` attribute refers to the data loaded
from ZCollection, not to positions in an external array. The two
datasets therefore need to be aligned separately, using ``time``
in this example.

.. jupyter-execute::

    import dask.array
    import zarr

    weights_path = str(root / 'weights.zarr')
    zarr.create_array(weights_path, data=numpy.arange(1.0, 10.0), chunks=(3, ))
    weights = dask.array.from_zarr(weights_path)

    def weighted_mean(ds, weights):
        first_day = ds['time'].values[0].astype('datetime64[D]')
        start = int((first_day - origin) / numpy.timedelta64(1, 'D'))
        stop = start + ds.dimensions['sample']
        selected = weights[start:stop].compute()
        return float(numpy.average(ds['signal'].values, weights=selected))

    collection.map(weighted_mean, weights,
                   selected_variables=('time', 'signal'),
                   delayed=False).compute(scheduler='synchronous')

Only the chunks intersecting the selected slice are read. For segment 3,
the function selects ``weights[5:9]``, which spans two chunks of the
external array.

For working with persisted Dask collections or scattered objects, see
:py:meth:`distributed.Client.persist` and
:py:meth:`distributed.Client.scatter`.


Choosing NumPy or Dask arrays
-----------------------------

The ``delayed`` parameter controls how variables are loaded before
being passed to the function.

- ``delayed=True`` (the default for ``map`` and ``map_overlap``) provides
  Dask arrays. Accessing ``ds['signal'].array`` returns a lazy array,
  while ``ds['signal'].values`` computes its values.

- ``delayed=False`` loads the variables into NumPy arrays before the
  function is called. Each call therefore requires enough memory to
  hold the partition or overlap window, along with the computed result.
