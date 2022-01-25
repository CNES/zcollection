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

Testing
-------

To run the test suite after installing the library, install (via pypi or
conda) `pytest <https://pytest.org>`__ and run ``pytest`` in the root
directory of the cloned repository.
