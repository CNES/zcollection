# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib.metadata
import pathlib
import sys

HERE = pathlib.Path(__file__).absolute().parent

# Insert the project root dir as the first element in the PYTHONPATH.
sys.path.insert(0, str(HERE.parent.parent))

# -- Project information -----------------------------------------------------

project = 'zcollection'
copyright = '(2022, CNES/CLS)'
author = 'CLS'

# The full version, including alpha/beta/rc tags
try:
    release = importlib.metadata.version(project)
except importlib.metadata.PackageNotFoundError:
    release = '0.0.0'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    'sphinx_inline_tabs',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True

autodoc_typehints = 'description'
autodoc_type_aliases = dict(
    ArrayLike='ArrayLike',
    DTypeLike='DTypeLike',
    Indexer='Indexer',
    PartitionCallback='PartitionCallback',
    QueryDict='QueryDict',
    Scalar='Scalar',
)

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = 'ZCollection'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = {
    'dask': ('https://docs.dask.org/en/latest/', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable', None),
}

# -- Extension configuration -------------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': [HERE.joinpath('examples')],
    'filename_pattern': r'[\\\/]ex_',
    'pypandoc': False,
}
