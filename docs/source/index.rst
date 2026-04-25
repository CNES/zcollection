ZCollection
===========

This project is a Python library manipulating data split into a
:py:class:`collection <zcollection.Collection>` of groups stored in
`Zarr v3 format <https://zarr.readthedocs.io/>`_.

A collection divides a dataset into partitions to make incremental
acquisitions or per-product updates cheap. Built-in partitionings are
:py:class:`by date <zcollection.partitioning.Date>`,
:py:class:`by sequence <zcollection.partitioning.Sequence>`, and
:py:class:`grouped sequences <zcollection.partitioning.GroupedSequence>`.

A collection partitioned by date with a monthly resolution looks like
this on disk::

   collection/
   ├── zarr.json
   ├── _zcollection.json
   ├── _catalog/                       # optional partition index
   │   ├── zarr.json
   │   └── c/0
   ├── _immutable/                     # non-partitioned variables
   │   └── zarr.json
   └── year=2024/
       └── month=03/
           ├── zarr.json
           ├── time/
           │   ├── zarr.json
           │   └── c/0
           └── ssh/
               ├── zarr.json
               └── c/0/0

Inserts can either overwrite existing partitions or merge with them
through pluggable :py:mod:`strategies <zcollection.collection.merge>`.

Storage backends are selected by URL scheme:

* ``file://`` — local filesystem
* ``memory://`` — in-process (tests, prototyping)
* ``s3://`` — object storage via `obstore`_ or `fsspec`_
* ``icechunk://`` — transactional Zarr v3 via `Icechunk`_

`Dask <https://dask.org/>`_ is used to scale operations over partitions.
The implementation is async; the sync API is a thin wrapper, and an
:py:mod:`zcollection.aio` mirror is published for async callers.

Views layered on top of a read-only base collection let you add or
recompute variables without touching the base.

.. _obstore: https://developmentseed.org/obstore/
.. _fsspec: https://filesystem-spec.readthedocs.io/
.. _Icechunk: https://icechunk.io/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   auto_examples/index.rst
   migration
   api
   release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
