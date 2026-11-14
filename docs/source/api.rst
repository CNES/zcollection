:tocdepth: 2

API reference
#############

Collections and views
=====================

.. autosummary::
   :toctree: _generated/

   zcollection.collection
   zcollection.collection.abc
   zcollection.collection.callable_objects.MapCallable
   zcollection.collection.callable_objects.PartitionCallable
   zcollection.collection.callable_objects.UpdateCallable
   zcollection.PartitionInfo
   zcollection.view
   zcollection.view.ViewReference

Creating and opening collections and views
==========================================

.. autosummary::
   :toctree: _generated/

   zcollection.create_collection
   zcollection.create_view
   zcollection.open_collection
   zcollection.open_view

Datasets and variables
======================

.. autosummary::
   :toctree: _generated/

   zcollection.dataset
   zcollection.meta
   zcollection.expression
   zcollection.variable.abc.Variable
   zcollection.variable.array
   zcollection.variable.delayed_array

Partitioning
============

.. autosummary::
   :toctree: _generated/

   zcollection.partitioning
   zcollection.partitioning.abc
   zcollection.partitioning.date
   zcollection.partitioning.sequence

.. _merging_datasets:

Merging datasets
================

.. autosummary::
   :toctree: _generated/

   zcollection.merging
   zcollection.merging.time_series
   zcollection.merging.period

Indexing
========

.. autosummary::
   :toctree: _generated/

   zcollection.indexing
   zcollection.indexing.abc

Synchronization
===============

.. autosummary::
   :toctree: _generated/

   zcollection.sync

Migration
=========

.. autosummary::
   :toctree: _generated/

   zcollection.migration.Outcome
   zcollection.migration.Status
   zcollection.migration.check
   zcollection.migration.check_view
   zcollection.migration.migrate
   zcollection.migration.migrate_view
   zcollection.migration.prune
   zcollection.migration.prune_view
   zcollection.migration.status
   zcollection.migration.view_status

Exceptions
==========

.. autosummary::
   :toctree: _generated/

   zcollection.AbsentPartition
   zcollection.ModifiedVariableError
   zcollection.PartitionErrors
