Release notes
=============

2026.x.0
--------

Features
^^^^^^^^
* A collection is usable without a Dask cluster (PR #16).
* ``load`` and ``update`` take a ``selected_partitions`` parameter.
* ``View.update`` takes a ``variables`` parameter (PR #14, issue #13).
* ``GroupedSequence`` partitioning, with the starting value of the group as a
  parameter.
* Immutable variables are insertable and updatable, and a collection holding
  only immutable variables is readable.
* A view holding no declared variable reads the collection it references.
* The metadata of a collection and of a dataset carry the dimensions, and the
  configuration of a partitioning carries the data types.
* ``VariableSchema``, ``DatasetSchema`` and ``Dimension`` are reachable from
  the package.

Storage
^^^^^^^
* Collections are written in the Zarr 3 format. A group and an array each hold
  their metadata in a ``zarr.json``, and an array names the file holding a
  chunk with the ``c`` prefix and the indices of the chunk, joined with a dot.
  A chunk whose values all equal the fill value of its array is not written,
  and reads back as that value.
* ``.zcollection`` carries the version of the layout it is written in. A
  collection written in a newer one is refused.
* A collection written by an earlier release opens read only, and writing to
  it is refused. Copying it into a new collection is what makes its data
  writable again, and a view written by an earlier release opens only once it
  has been copied.
* ``zcollection.migrate`` copies a collection written by an earlier release
  into a new one, partition by partition, with a filter selecting the
  partitions to copy.
  ``zcollection.status`` reports how far the copy has got,
  ``zcollection.check`` compares the values the two collections hold,
  ``zcollection.prune`` removes from the source the partitions the target
  holds, and ``zcollection.migrate_view``, ``zcollection.check_view`` and
  ``zcollection.view_status`` do the same for a view. A partition the target holds in full is not copied
  again, so a migration that stopped partway is resumed by calling the
  function again.
* The fill value of a variable is held in the ``_FillValue`` attribute of an
  array, beside the fill value Zarr 3 requires of every array.
* The codecs a variable names with the classes of numcodecs are translated into
  the ones Zarr 3 builds an array from.

Behaviour
^^^^^^^^^
* A partition is written with the compressor, the filters and the fill value
  the collection declares for each of its variables.
* The variables of a dataset read from a collection are ordered by name.
* ``insert`` returns the full path of each partition it wrote.
* A variable added to a dataset with the data it holds keeps its values.
* A write that fails carries a note naming the variable it was writing.
* Writing a partition stops with an error naming the variables the store does
  not list under it.
* Concatenating datasets that do not hold the same variables stops with an
  error naming the variables in disagreement.
* ``update`` passes the parameters it is given through (PR #15, issue #12).
* ``view.sync()`` honours ``filters`` (issue #24).

Removed
^^^^^^^
* ``new_variable``. ``Variable.duplicate`` builds a variable from another
  one, with the data and the properties it is given.

Renamed
^^^^^^^
* ``meta.Variable`` is named ``meta.VariableSchema``, and ``meta.Dataset`` is
  named ``meta.DatasetSchema``.
* ``ReadOnlyCollection.immutable`` is named ``immutable_path``.
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
