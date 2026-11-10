Computing on partitions
=======================

:py:meth:`~zcollection.collection.Collection.map` calls a function on each
partition and returns a Dask bag of the results.
:py:meth:`~zcollection.collection.Collection.map_overlap` does the same with
the neighboring partitions loaded around each one. Both read the collection
and write nothing; :doc:`updating` writes the results of a function to a
variable. Views provide the same methods.

The examples use consecutive daily measurements split into segments of three,
two and four samples.

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

The function receives the dataset of one partition. Computing the bag gives
one pair per partition: the keys of the partition and the result of the
function.

.. jupyter-execute::

    def mean(ds):
        return float(ds['signal'].values.mean())

    bag = collection.map(mean, delayed=False)
    bag.compute(scheduler='synchronous')

With ``delayed=False``, the variables are loaded into NumPy arrays before the
call; see `Choosing NumPy or Dask arrays`_. The bag is computed by the
scheduler given to ``compute``, here the calling process. Without one, it uses
the cluster of the current client, if there is one, or the default scheduler
of Dask bags.

Choosing the inputs
-------------------

``selected_variables`` lists the variables to load: the ones the function
reads. ``filters`` selects the partitions by their keys, and
``selected_partitions`` selects them by their paths. Given both, only the
partitions they both select are kept.

.. jupyter-execute::

    bag = collection.map(mean, selected_variables=('signal', ),
                         filters='segment >= 2', delayed=False)
    bag.compute(scheduler='synchronous')

.. jupyter-execute::

    bag = collection.map(mean, selected_variables=('signal', ),
                         selected_partitions=['segment=1', 'segment=3'],
                         delayed=False)
    bag.compute(scheduler='synchronous')

The arguments following the function, and the extra keyword arguments, are
passed to it:

.. jupyter-execute::

    def scaled_mean(ds, scale):
        return float(ds['signal'].values.mean() * scale)

    collection.map(scaled_mean, 2.0, selected_variables=('signal', ),
                   delayed=False).compute(scheduler='synchronous')

Knowing which partition is processed
------------------------------------

A function declaring a ``partition_info`` keyword parameter receives a
:py:class:`~zcollection.PartitionInfo` for the partition it processes.

.. jupyter-execute::

    def describe(ds, *, partition_info):
        return partition_info.name, partition_info.partition

    collection.map(describe, selected_variables=('signal', ),
                   delayed=False).compute(scheduler='synchronous')

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

See :doc:`partitioning` for the rules defining the keys and names.

Using neighboring partitions
----------------------------

``map_overlap`` loads each partition with ``depth`` neighboring partitions on
each side, concatenated along the partitioned dimension. ``filters`` and
``selected_partitions`` select the partitions to process; the neighbors of
each are loaded whether these parameters select them or not.
``partition_info.indices`` locates the partition in its window. At
``depth=1``:

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

The three-point average below reads samples of the neighboring segments,
then keeps the values of the processed partition with
``partition_info.indices``. It pads the two ends of the signal with zeros.

.. jupyter-execute::

    kernel = numpy.full(3, 1 / 3)

    def smooth(ds, *, partition_info):
        window = numpy.convolve(ds['signal'].values, kernel, mode='same')
        return window[partition_info.indices].round(2).tolist()

    collection.map_overlap(smooth, depth=1, selected_variables=('signal', ),
                           delayed=False).compute(scheduler='synchronous')

``indices`` keeps this meaning when the selected variables omit ``sample``,
for example when only an immutable variable is selected.

Reading an external Dask array
------------------------------

Dask handles the arguments passed to the function as
:py:meth:`dask.bag.Bag.map` does: a Dask array passed as an argument reaches
each call as the same lazy array. The function below selects the weights of
its days from an array with one value per day, and computes only that slice.
``partition_info.indices`` is relative to the data zcollection loads; an
external source needs its own alignment, here from ``time``.

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

Only the chunks a slice intersects are read: for segment 3, the function
selects ``weights[5:9]``, which intersects two chunks of this array. For
persisted Dask collections and scattered objects, see
:py:meth:`distributed.Client.persist` and
:py:meth:`distributed.Client.scatter`.

Choosing NumPy or Dask arrays
-----------------------------

``delayed`` sets what the function receives. With ``True``, the default for
``map`` and ``map_overlap``, ``ds['signal'].array`` is a Dask array and
``ds['signal'].values`` computes it. With ``False``, the variables are NumPy
arrays loaded before the call, and each call needs memory for the partition
or window and for its result.
