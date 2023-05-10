:tocdepth: 2

API Documentation
#################

Partitioning
============

Handles the partitioning of the collection.

.. autosummary::
  :toctree: _generated/

  zcollection.partitioning
  zcollection.partitioning.abc
  zcollection.partitioning.date
  zcollection.partitioning.registry
  zcollection.partitioning.sequence

.. _merging_datasets:

Merging of datasets
===================

Merging of existing datasets in a partition.

.. autosummary::
  :toctree: _generated/

  zcollection.merging
  zcollection.merging.time_series
  zcollection.merging.period

Variable
========

Variables handled by the datasets. These objects manage access to the data
stored in the collection.

.. autosummary::
  :toctree: _generated/

  zcollection.variable.abc
  zcollection.variable.array
  zcollection.variable.delayed_array

Collection
==========

.. autosummary::
  :toctree: _generated/

  zcollection.collection
  zcollection.dask_utils
  zcollection.dataset
  zcollection.expression
  zcollection.fs_utils
  zcollection.meta
  zcollection.sync
  zcollection.type_hints
  zcollection.view

Indexing
========

.. autosummary::
  :toctree: _generated/

  zcollection.indexing
  zcollection.indexing.abc

Convenience functions
=====================

.. autosummary::
  :toctree: _generated/

  zcollection.create_collection
  zcollection.create_view
  zcollection.open_collection
  zcollection.open_view
