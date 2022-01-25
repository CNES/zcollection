ZCollection
===========

This project is a Python library allowing manipulating data partitioned into a
**collection** of `Zarr <https://zarr.readthedocs.io/en/stable/>`_ groups.

This collection allows dividing a dataset into several partitions to facilitate
acquisitions or updates made from new products. Possible data partitioning is:
by **date** (hour, day, month, etc.) or by **sequence**.

A collection partitioned by date, with a monthly resolution, may look like on
the disk:

.. code-block:: ASCII

    collection/
    ├── year=2022
    │    ├── month=01/
    │    │    ├── time/
    │    │    │    ├── 0.0
    │    │    │    ├── .zarray
    │    │    │    └── .zattrs
    │    │    ├── var1/
    │    │    │    ├── 0.0
    │    │    │    ├── .zarray
    │    │    │    └── .zattrs
    │    │    ├── .zattrs
    │    │    ├── .zgroup
    │    │    └── .zmetadata
    │    └── month=02/
    │         ├── time/
    │         │    ├── 0.0
    │         │    ├── .zarray
    │         │    └── .zattrs
    │         ├── var1/
    │         │    ├── 0.0
    │         │    ├── .zarray
    │         │    └── .zattrs
    │         ├── .zattrs
    │         ├── .zgroup
    │         └── .zmetadata
    └── .zcollection

Partition updates can be set to overwrite existing data with new ones or to
update them using different **strategies**.

The `Dask library <https://dask.org/>`_ handles the data to scale the treatments
quickly.

It is possible to create views on a reference collection, to add and modify
variables contained in a reference collection, accessible in reading only.

This library can store data on POSIX, S3, or any other file system supported by
the Python library `fsspec
<https://filesystem-spec.readthedocs.io/en/latest/>`_. Note, however, only POSIX
and S3 file systems have been tested.
