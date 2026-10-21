ZCollection
===========

ZCollection stores a dataset too large for one file as a **collection**: a
tree of partitions, each holding a slice of the data in the
`Zarr format <https://zarr.readthedocs.io/en/stable/>`_. The collection is
read and written as a single dataset.

Adding a month of measurements to a ten-year collection writes one partition and
leaves the rest as it is. A partition takes its name from the values that put
the data there: by **date** (hour, day, month, year), or by **sequence** of
variables.

A collection partitioned by month, holding a longitude that is the same in
every partition:

.. code-block:: text

    collection/
    ├── .zcollection
    ├── .immutable/
    │    ├── lon/
    │    │    ├── c/
    │    │    │    └── 0
    │    │    └── zarr.json
    │    └── zarr.json
    └── year=2022/
         ├── month=01/
         │    ├── time/
         │    │    ├── c/
         │    │    │    └── 0
         │    │    └── zarr.json
         │    ├── var1/
         │    │    ├── c/
         │    │    │    └── 0/
         │    │    │         └── 0
         │    │    └── zarr.json
         │    └── zarr.json
         └── month=02/
              └── ...

A collection is read and written on a `Dask <https://dask.org/>`_ cluster, or
in the process that calls it, and is stored wherever the
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ library reaches:
a local disk, an S3 bucket, and the other file systems it implements.

Installation
------------

.. code-block:: shell

    conda install -c conda-forge zcollection

Documentation
-------------

The documentation holds the examples, the API and the release notes:
https://zcollection.readthedocs.io/en/latest/
