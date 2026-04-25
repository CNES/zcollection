Installation
============

zcollection 2.x is a Zarr v3-native rewrite of the library. It targets
modern Python and the modern scientific stack; there is no Zarr v2
compatibility runtime.

Requirements
------------

- Python **3.14** or later
- `zarr <https://zarr.readthedocs.io/>`_ ≥ 3.1.6
- `numpy <https://numpy.org/>`_ ≥ 1.20
- `numcodecs <https://numcodecs.readthedocs.io/>`_ ≥ 0.13
- `xarray <https://xarray.dev/>`_ ≥ 2024.10
- `pandas <https://pandas.pydata.org/>`_
- `dask <https://www.dask.org/>`_ ≥ 2022.8 and
  `distributed <https://distributed.dask.org/>`_
- `fsspec <https://filesystem-spec.readthedocs.io/>`_

Optional extras
~~~~~~~~~~~~~~~

- `pyarrow <https://arrow.apache.org/docs/python/>`_ — required by the
  Parquet-backed :class:`~zcollection.indexing.parquet.Indexer`.
- `obstore <https://developmentseed.org/obstore/>`_ ≥ 0.5 — recommended
  S3 backend (``s3://`` URLs). Install with::

      pip install "zcollection[obstore]"

- `icechunk <https://icechunk.io/>`_ ≥ 2.0 — transactional, multi-writer
  store (``icechunk://`` URLs)::

      pip install "zcollection[icechunk]"

- `s3fs <https://github.com/fsspec/s3fs/>`_ — alternative S3 backend
  through fsspec, used by the test suite when running against a local
  ``minio`` server.

Install from PyPI
-----------------

::

    pip install zcollection

To pull in the optional backends at the same time::

    pip install "zcollection[obstore,icechunk]"

Install from conda-forge
------------------------

::

    conda install -c conda-forge zcollection

Install from source
-------------------

Clone the repository and install in editable mode::

    git clone https://github.com/CNES/zcollection.git
    cd zcollection
    pip install -e ".[test]"

The ``test`` extra adds ``pytest``, ``pytest-cov`` and
``pytest-asyncio``.

Running the tests
-----------------

From the repository root::

    pytest zcollection/tests -q

No project-specific pytest options are required. Cloud-store tests are
opt-in and described in their own modules.

Building the documentation
--------------------------

The docs use Sphinx with Google-style docstrings. From the ``docs``
directory::

    make html

The rendered HTML lands in ``docs/build/html``.
