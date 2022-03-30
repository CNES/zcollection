Installation
============

Required dependencies
---------------------

- Python (3.8 or later)
- setuptools
- `dask <https://dask.pydata.org/>`_
- `distributed <https://distributed.dask.org/en/stable/>`_
- `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_
- `numcodecs <https://numcodecs.readthedocs.io/en/stable/>`_
- `numpy <https://numpy.org/>`_
- `pyarrow <https://arrow.apache.org/docs/python/>`_
- `xarray <http://xarray.pydata.org/en/stable/>`_
- `zarr <https://zarr.readthedocs.io/en/stable/>`_

.. note::

    `pyarrow` is optional, but required if you want to use the indexing API.

Instructions
------------

Installation via conda and sources
##################################

It is possible to install the latest version from source. First, install the
dependencies using conda::

    $ conda install dask distributed fsspec numcodecs numpy pandas pyarrow xarray zarr

Then, clone the repository::

    $ git clone git@github.com:CNES/zcollection.git
    $ cd zcollection

Finally, install the library using pip (it is possible to checkout a different
branch before installing)::

    $ pip install .

Installation via pip
####################

    $ pip install zcollection

Testing
-------

To run the test suite after installing the library, install (via pypi or
conda) `pytest <https://pytest.org>`__ and run ``pytest`` in the root
directory of the cloned repository.

The unit test process can be modified using options implemented for this
project, in addition to the options provided by ``pytest``. The available user
options are:

- **s3**: Enable tests on the local S3 server driven by minio. (default: False)
- **memory**: Use a file system in memory instead of the local file system.
  (default: False)
- **threads_per_worker**: Number of threads for each worker Dask.
  (default: the number of logical cores of the target platform).
- **n_workers**: Number of core for each worker Dask.
  (default: the number of cores of the target platform).

To run the tests using a local S3 server, driven by the ``minio`` software,
it's necessary to install the following optional requirements:

- `s3fs <https://github.com/fsspec/s3fs/>`_
- `requests <https://docs.python-requests.org/en/latest/>`_

You will need to install the ``minio`` program. You can find more information
on this web `page <https://min.io/download>`_.

Documentation
-------------

The documentation use sphinx and Google-style docstrings. To build the
documentation, run ``make html`` in the ``docs`` directory.
