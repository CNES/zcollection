ZCollection
===========

A Python library for manipulating data split into a **collection** of
`Zarr v3 <https://zarr.readthedocs.io/>`_ groups.

A collection divides a dataset into partitions to make incremental
acquisitions or per-product updates cheap. Built-in partitionings split
**by date** (hour, day, month, …), **by sequence**, or by **grouped
sequence**.

A :py:class:`~zcollection.Dataset` is a hierarchical container: it is a
root :py:class:`~zcollection.Group` that owns variables and attributes
directly and may also contain nested child groups, mirroring the native
Zarr v3 group hierarchy. Each child group becomes a real Zarr subgroup
on disk and round-trips transparently through partition I/O.

A collection partitioned by date with a monthly resolution looks like
this on disk:

.. code-block:: text

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

When the schema declares nested groups (e.g. ``/data_01/ku/...``), each
group materialises as a real Zarr v3 subgroup inside every partition:

.. code-block:: text

    collection/
    └── year=2024/month=03/
        ├── zarr.json
        ├── time/{zarr.json,c/0}
        └── data_01/
            ├── zarr.json
            └── ku/
                ├── zarr.json
                └── power/{zarr.json,c/0/0}

Inserts can either overwrite existing partitions or merge with them
through pluggable **strategies** (``replace``, ``concat``,
``time_series``, ``upsert``).

Storage backends are selected by URL scheme:

* ``file://`` — local filesystem
* ``memory://`` — in-process (tests, prototyping)
* ``s3://`` — object storage via `obstore
  <https://developmentseed.org/obstore/>`_ or `fsspec
  <https://filesystem-spec.readthedocs.io/>`_
* ``icechunk://`` — transactional Zarr v3 via `Icechunk
  <https://icechunk.io/>`_

`Dask <https://dask.org/>`_ is used to scale operations over partitions.
The implementation is async-first; the sync API is a thin wrapper, and
an :py:mod:`zcollection.aio` mirror is published for async callers.

Views layered on top of a read-only base collection let you add or
recompute variables without touching the base.

Quick start
-----------

.. code-block:: python

    import numpy
    import zcollection as zc

    # Build a (hierarchical) schema
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable(
            "ssh",
            dtype="float32",
            dimensions=("time", "x_ac"),
            fill_value=numpy.float32("nan"),
        )
        .with_group("/data_01/ku", attrs={"band": "Ku"})
        .with_dimension("range", size=240, chunks=240, group="/data_01/ku")
        .with_variable(
            "power",
            dtype="float32",
            dimensions=("time", "range"),  # ``time`` inherited from root
            group="/data_01/ku",
        )
        .build()
    )

    # Create a partitioned collection
    col = zc.create_collection(
        "file:///data/altimetry",
        schema=schema,
        axis="time",
        partitioning=zc.partitioning.Date(("time",), resolution="M"),
    )

    # Insert data and read it back
    col.insert(dataset)
    full = zc.open_collection("file:///data/altimetry", mode="r").query()
    print(full)  # multi-line, size-aware xarray-like repr
