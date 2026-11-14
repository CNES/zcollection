Installation
============

Required dependencies
---------------------

- Python (3.12 or later)
- `dask <https://dask.pydata.org/>`_ (2022.8.0 or later)
- `distributed <https://distributed.dask.org/en/stable/>`_
- `fasteners <https://fasteners.readthedocs.io/en/latest/>`_
- `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ (2024.12.0 or
  later)
- `numcodecs <https://numcodecs.readthedocs.io/en/stable/>`_
- `numpy <https://numpy.org/>`_ (1.20 or later)
- `xarray <http://xarray.pydata.org/en/stable/>`_
- `zarr <https://zarr.readthedocs.io/en/stable/>`_ (3.4 or later)

Optional dependencies
---------------------

- `pyarrow <https://arrow.apache.org/docs/python/>`_, required for the
  indexing API.


Installing ZCollection
----------------------

.. tab-set::

   .. tab-item:: conda

      The simplest way to install ZCollection is through conda-forge::

          $ conda install -c conda-forge zcollection

   .. tab-item:: pip

      ZCollection is available on PyPI::

          $ pip install zcollection

   .. tab-item:: From source

      To install ZCollection from source, first install its dependencies
      using conda::

          $ conda install -c conda-forge dask distributed fasteners fsspec numcodecs numpy xarray zarr

      Include ``pyarrow`` if you plan to use the indexing API.

      Clone the repository and navigate to its directory::

          $ git clone git@github.com:CNES/zcollection.git
          $ cd zcollection

      Check out a different branch first if needed, then install the
      library::

          $ pip install .


Testing
-------

ZCollection uses `pytest <https://pytest.org>`_ for testing. To run the test
suite, install the test dependencies::

    $ pip install '.[test]'

This installs ``pytest`` and ``pytest-cov``. Alternatively, you can install
``pytest`` separately using pip or conda.

From the root of the cloned repository, run::

    $ pytest

The test suite supports the following options in addition to the standard
pytest options:

- ``--s3``: Run tests against a local S3 server using MinIO
  (default: ``False``).
- ``--memory``: Use an in-memory file system instead of the local file system
  (default: ``False``).
- ``--threads_per_worker``: Number of threads per Dask worker
  (default: chosen by Dask based on the available CPU cores).
- ``--n_workers``: Number of Dask workers
  (default: chosen by Dask based on the available CPU cores).
- ``--processes``: Run Dask workers in separate processes instead of threads
  (default: ``False``).

To run tests against a local S3 server, install the following additional
dependencies:

- `s3fs <https://github.com/fsspec/s3fs/>`_
- `requests <https://docs.python-requests.org/en/latest/>`_

You also need the ``minio`` command. See the
`MinIO download page <https://min.io/download>`_ for installation instructions.


Documentation
-------------

ZCollection's documentation is built with Sphinx and uses Google-style
docstrings.

The ``conda/environment.yml`` file defines the environment needed to build
the documentation, including Sphinx and its extensions. Create the
environment with::

    $ conda env create -f conda/environment.yml

Then, from the ``docs`` directory, build the HTML documentation::

    $ make html
