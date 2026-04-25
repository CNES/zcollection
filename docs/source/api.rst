:tocdepth: 2

API Documentation
#################

Top-level package
=================

.. autosummary::
  :toctree: _generated/

  zcollection
  zcollection.api
  zcollection.aio
  zcollection.config
  zcollection.errors
  zcollection.types

Schema
======

The schema is the immutable description of the dataset: dimensions,
variables, attributes, and the format version. Build one with the
fluent :py:class:`~zcollection.Schema` builder.

.. autosummary::
  :toctree: _generated/

  zcollection.schema
  zcollection.schema.dimension
  zcollection.schema.variable
  zcollection.schema.attribute
  zcollection.schema.dataset
  zcollection.schema.builder
  zcollection.schema.serde
  zcollection.schema.versioning

Data containers
===============

In-memory dataset and variable objects, with bridges to and from xarray.

.. autosummary::
  :toctree: _generated/

  zcollection.data
  zcollection.data.variable
  zcollection.data.dataset

Stores
======

URL-driven backends: local filesystem, memory, fsspec, obstore, and
Icechunk.

.. autosummary::
  :toctree: _generated/

  zcollection.store
  zcollection.store.base
  zcollection.store.factory
  zcollection.store.local
  zcollection.store.memory
  zcollection.store.obstore_store
  zcollection.store.icechunk_store
  zcollection.store.layout

Codecs
======

Default codec profiles and shard sizing.

.. autosummary::
  :toctree: _generated/

  zcollection.codecs
  zcollection.codecs.defaults
  zcollection.codecs.sharding

Partitioning
============

.. autosummary::
  :toctree: _generated/

  zcollection.partitioning
  zcollection.partitioning.base
  zcollection.partitioning.date
  zcollection.partitioning.sequence
  zcollection.partitioning.grouped
  zcollection.partitioning.expression
  zcollection.partitioning.catalog

Collection
==========

.. autosummary::
  :toctree: _generated/

  zcollection.collection
  zcollection.collection.base
  zcollection.collection.merge

Views
=====

.. autosummary::
  :toctree: _generated/

  zcollection.view
  zcollection.view.base

Indexing
========

.. autosummary::
  :toctree: _generated/

  zcollection.indexing
  zcollection.indexing.parquet

I/O pipeline
============

.. autosummary::
  :toctree: _generated/

  zcollection.io

Dask integration
================

.. autosummary::
  :toctree: _generated/

  zcollection.dask
