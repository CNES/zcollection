Release notes
=============

2026.x.0
--------

Features
^^^^^^^^
* A collection is usable without a Dask cluster (PR #16).
* ``partitions``, ``load``, ``update``, ``map`` and ``map_overlap`` take a
  ``selected_partitions`` parameter. A path that follows the partitioning
  but names no existing partition is left out, and the collection or the
  view logs a warning naming it.
* ``insert`` takes a ``delayed`` parameter setting the representation its
  merge callable receives the partitions in.
* ``Collection.update_from`` and ``View.update_from`` write the variables of a
  dataset into existing partitions. Each row of the dataset is written to the
  row of its partition that has the same value of ``align_on``; a view reads
  ``align_on`` from its reference collection. The other rows and variables
  keep their values. ``update_from`` returns the paths of the partitions it
  wrote, relative to the collection or the view.
* ``View.validate_partitions`` lists the invalid partitions of a view. With
  ``fix``, the view writes again the ones the reference collection still has,
  and deletes the others.
* ``View.update`` takes a ``variables`` parameter (PR #14, issue #13).
* ``View.is_synced`` takes a ``filters`` parameter to check selected reference
  partitions. By default, it checks all reference partitions.
* ``GroupedSequence`` partitioning, with the starting value of the group as a
  parameter.
* Immutable variables are insertable and updatable, and a collection holding
  only immutable variables is readable.
* A view holding no declared variable reads and maps the collection it
  references with ``load``, ``map`` and ``map_overlap``.
* The metadata of a collection and of a dataset carry the dimensions, and the
  configuration of a partitioning carries the data types.
* ``VariableSchema``, ``DatasetSchema`` and ``Dimension`` are reachable from
  the package.
* ``AbsentPartition``, ``ModifiedVariableError`` and ``PartitionErrors`` are
  reachable from the package.

Storage
^^^^^^^
* Collections and views use Zarr 3. Existing Zarr 2 collections and views
  remain readable; migration creates a writable copy. See :doc:`migration`.
* ``.zcollection`` carries the version of the layout it is written in. A
  collection written in a newer one is refused.
* ``zcollection.migration`` supports selective migration of collections
  and views, progress reporting, data comparison and pruning.
  Interrupted migrations can be resumed.
* Collections and views maintain a partition index. If the index lists
  missing partitions, ``load``, ``map``, ``map_overlap`` and ``update`` skip
  them with a warning. ``Collection.copy`` also skips such missing source
  partitions. ``Collection.reindex`` and ``View.reindex`` rebuild the index
  from the store when it needs to be repaired. See :doc:`partition_index`.
* The fill value of a variable is held in the ``_FillValue`` attribute of an
  array, beside the fill value Zarr 3 requires of every array.
* Variable schemas accept native Zarr 3 codecs and supported numcodecs codecs.
  The numcodecs codecs are translated into their Zarr 3 equivalents.
* Opening a group or an array of a Zarr 3 collection costs one request for its
  metadata. Reading a collection, updating it and reading a view send fewer
  requests to the store.

Execution
^^^^^^^^^
* ``insert``, ``update`` and ``drop_partitions`` process each partition in a
  task of its own, and ``update_from`` follows the same rules. With
  ``distributed=True``, at most ``npartitions`` tasks run at the same time,
  and a task starts when another one finishes; ``npartitions`` defaults to
  the number of Dask workers. With ``distributed=False``, the partitions are
  processed one at a time in the calling process. By default, earlier
  releases inserted one partition at a time, updated the partitions in one
  batch per worker, and removed all the selected partitions at once. See
  :doc:`failures`.
* If a partition fails, these operations raise ``PartitionErrors``. Its
  ``completed``, ``failed`` and ``not_attempted`` attributes give the
  partitions completed, the traceback of each failed partition and the
  partitions not attempted. An ``except`` clause naming the type of an error
  raised while a partition is processed no longer catches that error.
* With ``stop_on_error=True``, the default, an operation starts no partition
  after the first failure, and the partitions in progress finish; with
  ``False``, it starts all of them.
* These operations hold the lock of the synchronizer until their tasks
  finish, including after a failure. ``insert`` writes the immutable
  variables under the same lock as the partitions.
* ``update`` returns the paths of the partitions it wrote, relative to the
  collection or the view.
* After a failed ``update``, the file system cache of the selected partitions
  is cleared.
* ``merging.perform`` takes the arguments of the merge callable in
  ``merge_kwargs``. Every extra keyword argument of ``insert`` reaches the
  merge callable. One named after a positional parameter of the merge
  callable makes each partition fail with ``TypeError``.

Behaviour
^^^^^^^^^
* With ``depth``, ``map_overlap`` and ``update`` load the neighbors of a
  partition whether ``filters`` and ``selected_partitions`` select them or
  not, so a partial update
  writes the values a full one does (issue #27). Earlier releases formed the
  windows among the filtered partitions only.
* A keyword argument named ``selected_partitions`` given to ``update``,
  ``map`` or ``map_overlap`` no longer reaches the function.
* With ``ProcessSync``, ``Collection.is_locked()`` reports whether its
  synchronizer holds the lock. It returns ``False`` after release and
  when the lock is held by another process.
* A partition is written with the compressor, the filters and the fill value
  the collection declares for each of its variables.
* Dimensions other than the partitioned dimension must declare a size.
* A variable's fill value must be compatible with its data type and encoding.
* Missing values are masked correctly when a variable's fill value is
  NaN or NaT.
* Variables loaded from a collection have a deterministic order.
* ``insert`` returns partition paths relative to the collection.
* ``drop_partitions`` returns a tuple of the paths of the partitions it
  removed, relative to the collection. A partition the index lists but the
  store does not hold is left out.
* ``drop_partitions(timedelta=...)`` measures the age of a partition
  whatever the local time zone.
* Inserting an empty dataset with ``distributed=True`` returns an empty tuple.
* ``Collection.copy`` without ``filesystem`` writes the copy on the file system
  of the collection.
* ``insert`` hands its merge callable the representation of the dataset it
  receives. A dataset of Dask arrays stays lazy through an insertion made
  without a Dask cluster.
* When ``data`` is omitted, ``Dataset.add_variable`` uses the variable's
  existing data, if available.
* ``Dataset.add_variable`` raises ``ValueError`` when the number of data
  dimensions differs from the variable's declared dimensions.
* ``Sequence.decode`` raises ``ValueError`` when the number of values differs
  from the number of partitioning variables.
* Writing a partition stops with an error naming the variables the store does
  not list under it.
* Concatenating datasets that do not hold the same variables stops with an
  error naming the variables in disagreement.
* ``update`` forwards positional and keyword arguments to the user callback
  (PR #15, issue #12).
* Callbacks declaring ``partition_info`` receive a
  :py:class:`zcollection.PartitionInfo` object with ``partition``, ``name``,
  ``dimension`` and ``indices`` attributes. Callbacks that unpacked the former
  tuple must access ``dimension`` and ``indices`` through these attributes.
  The object is available in ``map``, ``map_overlap`` and ``update``, including
  calls without overlap and the inference call of ``update``.
* ``Collection.update`` and ``View.update`` default ``delayed`` to ``None``.
  Variables load as NumPy arrays with ``distributed=False`` and as Dask
  arrays with ``distributed=True``. An explicit ``delayed`` value overrides
  this choice.
* Operations with overlap accept selections that omit the partitioned
  dimension, including selections containing only immutable variables.
* Views support reference collections on a different filesystem when adding
  variables, synchronizing partitions and running callbacks.
* ``View.load`` returns ``None`` when an explicit partition selection retains
  no data, including when the reference holds immutable variables.
* ``view.sync()`` honours ``filters`` (issue #24).

Removed
^^^^^^^
* The ``Dataset`` constructor no longer accepts ``chunks`` or
  ``block_size_limit``. Configure chunk sizes in ``Dimension`` and the
  block-size limit in ``DatasetSchema``.
* The ``validate`` parameter has been removed from ``Collection.insert``.
  Compatibility checks for partitioned data can no longer be disabled.
* ``new_variable``. ``Variable.duplicate`` builds a variable from another
  one, with the data and the properties it is given.

Renamed
^^^^^^^
* The ``npartition`` parameter of ``Collection.map_overlap`` is named
  ``npartitions``.
* ``meta.Variable`` is named ``meta.VariableSchema``, and ``meta.Dataset`` is
  named ``meta.DatasetSchema``.
* ``Dataset.dims_chunk`` is named ``dim_chunks``, and holds a mapping of a
  dimension name to the size of its chunks. It was a property returning a
  sequence of ``Dimension``.

Requirements
^^^^^^^^^^^^
* Python 3.12 or later.
* Zarr 3.4 or later. The bounds on ``fsspec``, ``s3fs`` and ``numcodecs`` are
  lifted.

2026.6.0
--------
* A directory that does not match the partitioning scheme is skipped when the
  partitions of a collection are listed.
* Masked arrays are written with the versions of Dask that follow NumPy 2.
* The package publishes from a checkout that carries its tags.

2024.2.0
--------
* Refactor merging module and improve temporary directory handling.
* Bug fix: Add cache invalidation for updated partitions.

2024.1.0
--------
* Bug fix: Partitions must not be sorted lexicographically.

2023.11.2
---------
* Returns a list of added/deleted partitions.

2023.11.1
---------
* Skip filesystem scans: Queries utilizing an indexer reuse known partitions for
  efficiency.

2023.11.0
---------
* Evaluate an expression from a dataset.
* Fix linter warning.
* Validate partitions.

2023.10.0
---------
* Merge time series with data gaps.
* Fix Linux fork startup blocking test.
* Correction of issues related to CI executions.
* Added an option to specify the list of variables used by the callback
  function for updating a partition.
* Classmethods removed from indexers.

2023.5.0
--------
* Add missing copyrights.
* Modularise code to reduce the number of lines per module.
* Writing variables is limited to the worker being used.
* Improve test coverage.
* #9: Read the version attribute directly from the ``version.py`` module.
* #8: Incomplete overlaps with more than one worker.
* #7: Fix bug, in the update method, if the user has selected multiple
  partitions the selected variables must contain the updated variables.
* #6: the parameter name for specifying the number of concurrent inserts is
  incorrect.
* #3: Add a trim argument to the ``update`` method, like Dask's Dask's
  ``map_overlap``.
* Update the documentation.
* Refactor the code.
* Loading data using Dask or Numpy.
* Variable adds attributes to partitions.

2023.3.2
--------
* Writing a partition with many variables is slow.
* Writing metadata only in the collection's configuration.
* Adding an inter-process lock
* If a variable has been modified since its initialization, the library throws a
  specific exception to warn the user.

2023.3.1
--------
* Fixed a compatibility issue with fspec 2023.3.0.

2023.3.0
--------
* Apply an optional mask before querying an indexer.

2023.2.0
--------
* Synchronize the view with the reference collection.
* Support for Python 3.11.
* Bug fixes.
* Optimization of the insertion of new partitions.
* Copy collection over different file systems.
* Export Dataset to Zarr group.

2022.12.0/2022.12.1
-------------------

Release on December 2, 2022

* Write immutable variables of a dataset into a single group.
* Possibility to update partitions using neighbor partitions (useful for
  filtering, for example).
* Refactor methods overlapping partitions.
* Update documentation.

2022.10.2/2022.10.1
-------------------

Release on October 13, 20212

* Add compatibility with Python 3.8.

2022.10.0
---------

Release on October 7, 20212

* Added an option to the method ``drop_partitions`` to drop partitions
  older than a specified time delta relative to the current time.

2022.8.0
--------

Release on August 14, 2022

* Support Python starting 3.9.
* Refactor convenience functions.
* Refactor dataset & variables modules.
* The indexer can return only the partition keys.
* Optimization of dataset handling.
* Bug fixes.

0.2 / 2020-04-04
----------------

Release on April 4, 2020

* Installation from PyPi.
* Unsigned integers are not handled.

0.1 / 2022-08-30
-----------------

Release on March 30, 2020

* First public version.
