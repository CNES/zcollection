Release notes
=============

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
