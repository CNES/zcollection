Release notes
=============

2026.x.0
--------

New features
^^^^^^^^^^^^

* Collections can be read and written without a Dask cluster (:pr:`16`).
* ``partitions``, ``load``, ``update``, ``map``, and ``map_overlap`` accept
  ``selected_partitions`` to select partitions by path. A path that matches
  the partitioning scheme but does not exist is skipped with a warning naming
  it, for both collections and views.
* :py:meth:`.Collection.update_from` and :py:meth:`.View.update_from` update
  existing partitions using values from a dataset. Rows are matched by
  ``align_on``; views read this variable from their reference collection. Other
  rows and variables remain unchanged. The methods return the paths of the
  updated partitions, relative to the collection or view.
* :py:meth:`.View.validate_partitions` identifies invalid view partitions. With
  ``fix=True``, it rewrites partitions still present in the reference
  collection and removes the others.
* :py:meth:`.View.update` accepts a ``variables`` parameter (:pr:`14`,
  :issue:`13`).
* :py:meth:`.View.is_synced` accepts ``filters`` to check selected reference
  partitions. By default, it checks all reference partitions.
* The new :py:class:`.GroupedSequence` partitioning groups sequence values into
  bins with a configurable starting value.
* Collections support inserting and updating immutable variables, as well
  as reading collections that contain only immutable variables.
  :py:meth:`.Collection.add_variable` and :py:meth:`.Collection.drop_variable`
  also support immutable variables.
* :py:meth:`.Collection.add_dimension` declares a new dimension in the
  collection's configuration, while :py:meth:`.Collection.drop_dimension`
  removes one that no variable uses. Dimensions must be declared before
  they can be used with :py:meth:`.Collection.add_variable` (:pr:`23`).
* Views with no declared variables can use ``load``, ``map``, and
  ``map_overlap`` to access variables from their reference collection.
* Dataset and collection metadata include dimension definitions, and
  partitioning configurations include data types (:pr:`23`, :pr:`17`).
* :py:class:`.VariableSchema`, :py:class:`.DatasetSchema`,
  :py:class:`.Dimension`, :py:exc:`.AbsentPartition`,
  :py:exc:`~zcollection.ModifiedVariableError`, and
  :py:exc:`.PartitionErrors` are available directly from the package.

Storage and migration
^^^^^^^^^^^^^^^^^^^^^

* Collections and views now use Zarr 3. Existing Zarr 2 collections and views
  remain readable; migration creates a writable copy. See :doc:`migration`.
* The ``.zcollection`` configuration records the storage layout version.
  Collections written with a newer layout version cannot be opened.
* ``zcollection.migration`` supports selective migration of collections and
  views, progress reporting, data comparison, and pruning. Interrupted
  migrations can be resumed.
* Collections and views maintain a partition index. When the index contains
  partitions missing from storage, ``load``, ``map``, ``map_overlap``, and
  ``update`` skip them with a warning. :py:meth:`.Collection.copy` also skips
  missing source partitions. :py:meth:`.Collection.reindex` and
  :py:meth:`.View.reindex` rebuild the index from storage. See
  :doc:`partition_index`.
* Variable fill values are recorded in the ``_FillValue`` array attribute,
  alongside the fill value required by Zarr 3 for every array.
* Variable schemas support native Zarr 3 codecs and compatible numcodecs
  codecs. Supported numcodecs codecs are translated to their Zarr 3
  equivalents.
* Opening a Zarr 3 group or array requires one metadata request. Reading
  and updating collections, and reading views, now require fewer requests
  to storage.

Execution and failure handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``insert``, ``update``, ``update_from``, ``add_variable``,
  ``drop_variable``, ``drop_partitions``, ``validate_partitions``, ``copy``,
  ``View.sync``, ``View.is_synced``, and ``create_view`` process partitions
  independently.

  With ``distributed=True``, each partition runs in a separate Dask task.
  At most ``npartitions`` tasks are in flight, with another submitted as
  each task finishes. The default ``npartitions=None`` uses the number of
  Dask workers, except for ``insert``, which defaults to 1. With
  ``distributed=False``, partitions are processed sequentially in the
  calling process. Immutable variables are added and removed in the
  calling process.

  By default, earlier releases updated and copied partitions in one batch
  per worker, added or removed variables and synchronized views across
  all partitions at once, and removed all selected partitions at once.
  With ``npartitions=None``, insertion previously processed one partition
  at a time. See :doc:`failures`.

* Partition failures raise :py:exc:`.PartitionErrors`, which provides:

  - ``completed``: paths of completed partitions.
  - ``failed``: paths mapped to error tracebacks.
  - ``not_attempted``: paths of partitions not processed.

  After a failed ``add_variable``, ``completed`` names the partitions that
  received the array before cleanup. After a failed ``View.is_synced``, it
  names partitions found out of sync. After a failed check with
  ``validate_partitions``, it names invalid partitions found. Errors raised
  while processing a partition are reported through ``PartitionErrors``;
  catching their original exception type directly no longer handles them.

* These operations accept ``npartitions`` and ``stop_on_error``. With
  ``stop_on_error=True`` (the default), no new tasks are submitted after a
  failure; tasks already submitted to Dask finish. With ``False``, every
  partition is attempted.
* Operations hold their synchronizer lock until their tasks finish, even
  after a failure. ``View.is_synced`` and ``validate_partitions`` without
  ``fix`` are exceptions. ``insert`` writes immutable variables under the
  same lock as its partitions.
* ``update`` returns paths of written partitions relative to the collection
  or view. After a failed update, the filesystem cache of the selected
  partitions is cleared.
* :py:func:`.merging.perform` accepts merge-callback arguments through
  ``merge_kwargs``. Extra keyword arguments passed to ``insert`` are
  forwarded to the merge callable. Passing a keyword already supplied as
  one of the callable's positional parameters causes each affected
  partition to fail with ``TypeError``.

Behavior changes and fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Partition selection and processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* With ``depth``, ``map_overlap`` and ``update`` load neighboring partitions
  even when ``filters`` or ``selected_partitions`` exclude them. Partial
  updates therefore use the same windows as full updates. Earlier releases
  formed overlap windows only from selected partitions (:issue:`27`).
* The ``selected_partitions`` argument to ``update``, ``map``, and
  ``map_overlap`` is no longer passed to the user callback.
* Operations with overlap accept variable selections that omit the
  partitioned dimension, including selections containing only immutable
  variables.
* ``insert`` returns partition paths relative to the collection and returns
  an empty tuple when inserting an empty dataset with
  ``distributed=True``.
* ``drop_partitions`` returns a tuple of removed partition paths relative
  to the collection, excluding entries that exist in the index but not in
  storage. Its ``timedelta`` selection measures partition age independently
  of the local time zone.
* :py:meth:`.Collection.validate_partitions` returns a tuple of invalid
  partition paths relative to the collection. It also detects partitions
  missing an array for a declared variable spanning the partitioned dimension.

Variables and datasets
~~~~~~~~~~~~~~~~~~~~~~

* Partitions use the compressor, filters, and fill value declared for each
  variable in the collection.
* Dimensions other than the partitioned dimension must declare a size.
* Fill values must be compatible with the variable's data type and encoding.
  NaN and NaT fill values are now masked correctly.
* Variables loaded from a collection have a deterministic order.
* ``insert`` preserves the representation of its input dataset when passing
  it to the merge callable. Dask arrays remain lazy when inserting without
  a Dask cluster. The new ``delayed`` parameter lets callers control the
  representation passed to the merge callable.
* :py:meth:`.Dataset.add_variable` uses a variable's existing data, when
  available, if ``data`` is omitted. It raises ``ValueError`` if the number of
  data dimensions differs from the declared dimensions.
* :py:meth:`.Sequence.decode` raises ``ValueError`` if the number of values
  does not match the number of partitioning variables.
* Errors when writing a partition identify variables missing from its
  storage directory. Errors when concatenating datasets identify variables
  that differ between the datasets.

Callbacks and synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``update`` forwards additional positional and keyword arguments to the
  user callback (:pr:`15`, :issue:`12`).
* Callbacks declaring ``partition_info`` receive a
  :py:class:`zcollection.PartitionInfo` object with ``partition``, ``name``,
  ``dimension``, and ``indices`` attributes. Code that unpacked the former
  tuple must now access ``dimension`` and ``indices`` as attributes. The
  object is available in ``map``, ``map_overlap``, and ``update``, including
  calls without overlap and the output-inference call of ``update``.
* :py:meth:`.Collection.update` and :py:meth:`.View.update` default to
  ``delayed=None``. With ``distributed=False``, variables load as NumPy
  arrays; with ``distributed=True``, they load as Dask arrays. Explicit
  ``delayed`` values override this behavior.
* :py:meth:`.Collection.is_locked` reports whether its :py:class:`.ProcessSync`
  synchronizer holds the lock. It returns ``False`` after the lock is released
  and when another process holds it.

Views and recovery
~~~~~~~~~~~~~~~~~~

* If ``add_variable`` fails and its cleanup also fails, the variable
  remains declared in the schema. The error includes a note naming
  partitions where its array may remain.
* :py:meth:`.View.drop_variable` removes the schema entry only after all
  partitions have succeeded. Removing an already absent array succeeds,
  allowing a second call to complete a failed removal.
* :py:meth:`.View.add_variable` records the variable in the view configuration
  before creating its arrays, including when the view has no partitions.
* ``View.sync(filters=...)`` records its filters after all selected
  partitions have been synchronized. ``view.sync()`` honors ``filters``
  (:issue:`24`).
* :py:meth:`.View.is_synced` raises :py:exc:`.PartitionErrors` when a partition
  lacks its ``.checksum`` array and returns ``True`` when no partitions are
  selected.
* Views can use reference collections on a different file system when
  adding variables, synchronizing partitions, and running callbacks.
* :py:meth:`.View.load` returns ``None`` when an explicit partition selection
  contains no data, including when the reference collection has immutable
  variables.
* :py:meth:`.Collection.copy` uses the source collection's file system when
  ``filesystem`` is omitted. It rejects non-empty targets and copies
  immutable variables. The configuration is written before the partitions,
  so a partially completed copy can be opened with the partitions present
  in its target storage.

Removed APIs
^^^^^^^^^^^^

* The :py:class:`.Dataset` constructor no longer accepts ``chunks`` or
  ``block_size_limit``. Configure chunk sizes through :py:class:`.Dimension`
  and the block-size limit through :py:class:`.DatasetSchema`.
* :py:meth:`.Collection.insert` no longer accepts ``validate``. Compatibility
  checks for partitioned data can no longer be disabled.
* ``new_variable`` has been removed. Use :py:meth:`.Variable.duplicate` to
  create a variable from another, supplying the data and properties required.

Renamed APIs
^^^^^^^^^^^^

* :py:meth:`.Collection.map_overlap`: ``npartition`` is now ``npartitions``.
* :py:meth:`.View.drop_variable`: ``varname`` is now ``variable``.
* ``meta.Variable`` and ``meta.Dataset`` are now
  :py:class:`.meta.VariableSchema` and :py:class:`.meta.DatasetSchema`.
* ``Dataset.dims_chunk`` is now :py:attr:`~.Dataset.dim_chunks`, a mapping from
  dimension names to chunk sizes. Previously, it was a property returning a
  sequence of ``Dimension`` objects.

Requirements
^^^^^^^^^^^^

* Python 3.12 or later.
* Zarr 3.4 or later, and fsspec 2024.12.0 or later. The upper bounds on
  ``fsspec`` and ``numcodecs`` have been removed, and ``pandas`` is no
  longer required.


2026.6.0
--------

* Ignore directories that do not match the partitioning scheme when listing
  a collection's partitions.
* Fix writing masked arrays with Dask versions following NumPy 2.
* Support publishing the package from a checkout containing its tags.


2024.2.0
--------

* Refactor the merging module and improve temporary directory handling.
* Invalidate the cache after updating partitions.


2024.1.0
--------

* Fix partition ordering: partitions must not be sorted lexicographically.


2023.11.2
---------

* Return a list of added and removed partitions.


2023.11.1
---------

* Avoid filesystem scans for indexer queries by reusing known partitions.


2023.11.0
---------

* Support evaluating expressions from a dataset.
* Fix a linter warning.
* Add partition validation.


2023.10.0
---------

* Support merging time series containing gaps.
* Fix a test blocked by Linux process forking.
* Fix issues with CI execution.
* Allow callers to select the variables loaded for a partition update
  callback.
* Remove class methods from indexers.


2023.5.0
--------

* Add missing copyright notices.
* Reorganize modules to reduce their size.
* Limit variable writes to the worker processing them.
* Improve test coverage.
* Read the version attribute directly from ``version.py`` (:issue:`9`).
* Fix incomplete overlaps when using more than one worker (:issue:`8`).
* Fix updates with multiple selected partitions when the selected variables
  must contain the variables being updated (:issue:`7`).
* Correct the name of the parameter controlling concurrent inserts
  (:issue:`6`).
* Add ``trim`` to :py:meth:`~.Collection.update`, following Dask's
  ``map_overlap`` (:issue:`3`).
* Update the documentation and refactor the code.
* Support loading data with Dask or NumPy.
* Add variable attributes to partitions.


2023.3.2
--------

* Improve performance when writing partitions with many variables.
* Store metadata only in the collection configuration.
* Add an inter-process lock.
* Raise a dedicated exception when a variable has changed since its
  initialization.


2023.3.1
--------

* Fix compatibility with ``fsspec`` 2023.3.0.


2023.3.0
--------

* Support applying an optional mask before querying an indexer.


2023.2.0
--------

* Support synchronizing views with their reference collections.
* Add Python 3.11 support.
* Fix bugs and optimize insertion of new partitions.
* Support copying collections across file systems.
* Support exporting a dataset to a Zarr group.


2022.12.0 / 2022.12.1
---------------------

Released December 2, 2022.

* Store immutable dataset variables in a single group.
* Support updating partitions using neighboring partitions, for example
  when filtering data.
* Refactor the methods that handle overlapping partitions.
* Update the documentation.


2022.10.2 / 2022.10.1
---------------------

Released October 13, 2022.

* Add Python 3.8 compatibility.


2022.10.0
---------

Released October 7, 2022.

* Allow :py:meth:`~.Collection.drop_partitions` to remove partitions older than
  a specified duration relative to the current time.


2022.8.0
--------

Released August 14, 2022.

* Support Python 3.9 and later.
* Refactor convenience functions and the dataset and variable modules.
* Allow indexers to return only partition keys.
* Optimize dataset handling.
* Fix bugs.


0.2 / 2022-04-04
----------------

Released April 4, 2022.

* Support installation from PyPI.
* Unsigned integers are not supported.


0.1 / 2022-03-30
----------------

Released March 30, 2022.

* First public release.
