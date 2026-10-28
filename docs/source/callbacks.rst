Mapping and updating partitions
===============================

``Collection`` and ``View`` apply callbacks to partitions of a dataset. Use
``map`` to collect a result per partition, ``map_overlap`` when the callback
needs neighboring partitions, and ``update`` to write results to variables.

.. list-table:: Choosing an operation
   :header-rows: 1
   :widths: 18 28 26 28

   * - Method
     - Callback input
     - Callback result
     - Execution
   * - ``map``
     - One partition
     - Any value, including ``None``
     - Returns a lazy Dask Bag
   * - ``map_overlap``
     - A partition and its available neighbors
     - Any value, including ``None``
     - Returns a lazy Dask Bag
   * - ``update``
     - One partition, with neighbors when ``depth > 0``
     - A mapping of output variable names to arrays
     - Executes and writes before returning

Each item in a mapping Bag is ``(partition_scheme, callback_result)``. Calling
``compute()`` collects these results into a Python list. Use ``update`` for
partitioned array outputs you want to store in the collection.

For a view, inputs can come from its reference collection and its own
variables. An update writes only variables belonging to the view. A view with
no variables of its own can use ``map`` and ``map_overlap``.

A small collection
------------------

The examples on this page use consecutive daily measurements split into
segments of three, two and four samples. They load NumPy inputs and run in
the calling process.

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

Selecting inputs and identifying the partition
----------------------------------------------

Use ``selected_variables`` to load the inputs the callback needs. Arguments
following the callback, and additional keyword arguments, are passed to it.
A callback declaring ``partition_info`` receives a
:py:class:`~zcollection.PartitionInfo` object identifying the current partition.

.. jupyter-execute::

    def summarize(ds, scale, *, partition_info):
        return {
            'name': partition_info.name,
            'mean': float(ds['signal'].values.mean() * scale),
        }

    bag = collection.map(
        summarize, 2.0, selected_variables=('signal', ), delayed=False)
    summaries = bag.compute(scheduler='synchronous')
    [result for _, result in summaries]

``PartitionInfo`` provides four attributes:

.. list-table:: Callback context
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Meaning
   * - ``partition``
     - Mapping of partition keys to their values, such as ``{'segment': 2}``.
   * - ``name``
     - Relative name of the partition, usable with ``selected_partitions``.
   * - ``dimension``
     - Dimension along which the data is partitioned, ``sample`` here.
   * - ``indices``
     - Slice selecting the central partition within the input window.

You can use the keys to query another source and the name to select the same
partition in the collection. See :doc:`partitioning` for the rules defining
these keys and names. Without overlap, ``indices`` is ``slice(None)``.

Working with neighbors
----------------------

``depth`` counts neighboring partitions on each side among the selected
partitions. Filters are applied before the windows are formed. At ``depth=1``,
the three targets in this example receive these windows:

.. list-table:: Windows and central slices
   :header-rows: 1
   :widths: 16 30 24 30

   * - Target segment
     - Input segments
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

The central slice retains this meaning even when the selected variables omit
``sample``, for example when only an immutable variable is selected. With
``depth=0``, ``map_overlap`` processes one partition and provides
``slice(None)``.

The following three-point average uses samples from neighboring segments.
The example uses zero padding at the two ends of the complete signal.
``map_overlap`` returns whatever the callback produces, so select the central
slice in the callback when you want one result per original sample.

.. jupyter-execute::

    kernel = numpy.full(3, 1 / 3)

    def smooth_window(ds):
        return numpy.convolve(ds['signal'].values, kernel, mode='same')

    def smooth_partition(ds, *, partition_info):
        return smooth_window(ds)[partition_info.indices]

    mapped = collection.map_overlap(
        smooth_partition, depth=1,
        selected_variables=('signal', ), delayed=False)
    smoothed = mapped.compute(scheduler='synchronous')
    {
        collection.partitioning.join(partition, '/'): values.round(2).tolist()
        for partition, values in smoothed
    }

Writing the results
-------------------

Create the output variable before updating it. ``variables`` names the outputs
of ``update``; ``selected_variables`` still controls its inputs. The output
variable can therefore be omitted from the loaded dataset.

With overlap, ``trim=True`` extracts the central slice from each returned
array along its partitioned dimension. Return the result for the whole window
in this mode. Use ``trim=False`` when your callback already returns only the
central partition's values.

.. jupyter-execute::

    collection.add_variable(
        meta.VariableSchema(
            'smoothed', 'float64', dimensions=('sample', )),
        distributed=False)

    def write_smoothed(ds):
        return {'smoothed': smooth_window(ds)}

    collection.update(
        write_smoothed, variables=('smoothed', ),
        selected_variables=('signal', ), depth=1,
        delayed=False, distributed=False)
    loaded = collection.load(
        selected_variables=('smoothed', ), delayed=False, distributed=False)
    loaded['smoothed'].values.round(2)

If you omit ``variables``, zcollection infers the output names by calling the
callback on the first available selected partition. This extra call uses that
partition alone, even when updates use overlap. Its ``indices`` is
``slice(None)``, and its returned arrays are discarded. Supply ``variables``
when the callback performs expensive work or has side effects you want to
avoid during inference.

This example reads ``signal`` and writes ``smoothed``, so updates leave the
filter's inputs unchanged. A callback that reads and rewrites the same
variables must be idempotent and account for neighbors containing values
already written by earlier calls. Distributed callbacks can run in any order.

An indexed partition missing from the store is skipped with a warning when
it is read. A missing neighbor is excluded from the window. Errors from the
callback stop the operation; other partitions may already have been written
when an update fails.

Loading and execution choices
-----------------------------

``delayed`` controls the variables in the dataset passed to the callback.
With ``False``, they are loaded into NumPy arrays before the call. With
``True``, ``ds['signal'].array`` is a Dask array; accessing
``ds['signal'].values`` computes its NumPy values.

.. list-table:: Defaults and execution controls
   :header-rows: 1
   :widths: 24 25 51

   * - Method
     - Default ``delayed``
     - Execution control
   * - ``map``, ``map_overlap``
     - ``True``
     - Choose the scheduler when computing the returned Bag.
   * - ``update``
     - ``None``, following ``distributed``
     - ``distributed=False`` runs locally; ``True`` uses a Dask client.

``update`` defaults to ``distributed=True``. To run NumPy callbacks on Dask
workers, set ``delayed=False`` and ``distributed=True``. Each concurrent
callback needs memory for its selected partition or window and its results.
See :doc:`auto_examples/ex_without_dask` for operations in the calling process.

Reading part of an external Dask array
--------------------------------------

Dask manages external positional and keyword arguments to ``map`` and
``map_overlap`` according to :py:meth:`dask.bag.Bag.map`. ``delayed`` controls
the variables in the dataset supplied by zcollection.

The weights below have one value per day, with the same origin as ``time``.
The callback derives the external offset from that coordinate, selects its
weights and computes them. ``PartitionInfo.indices`` is relative to a
zcollection window; an external source needs its own coordinate alignment.

.. jupyter-execute::

    import dask.array
    import zarr

    weights_path = str(root / 'weights.zarr')
    zarr.create_array(
        weights_path, data=numpy.arange(1.0, 10.0), chunks=(3, ))
    weights = dask.array.from_zarr(weights_path)

    def weighted_mean(ds, weights):
        first_day = ds['time'].values[0].astype('datetime64[D]')
        start = int((first_day - origin) / numpy.timedelta64(1, 'D'))
        stop = start + ds.dimensions['sample']
        selected_weights = weights[start:stop].compute()
        return float(numpy.average(
            ds['signal'].values, weights=selected_weights))

    weighted = collection.map(
        weighted_mean, weights,
        selected_variables=('time', 'signal'), delayed=False)
    weighted.compute(scheduler='synchronous')

Only chunks intersecting a requested slice need to be read. Segment 3 selects
``weights[5:9]``, which intersects two chunks of this array.

For persisted collections and scattered objects, see
:py:meth:`distributed.Client.persist` and
:py:meth:`distributed.Client.scatter`.
