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
import datetime
import importlib.metadata
import os
import pathlib
import sys

from sphinx_gallery.sorting import FunctionSortKey

HERE = pathlib.Path(__file__).absolute().parent

# Insert the project root dir as the first element in the PYTHONPATH.
sys.path.insert(0, str(HERE.parent.parent))

# The Jupyter kernels that run the code of the pages import the project from
# the same directory.
os.environ['PYTHONPATH'] = os.pathsep.join(
    filter(None, (str(HERE.parent.parent), os.environ.get('PYTHONPATH'))))

# -- Project information -----------------------------------------------------

project = 'zcollection'
copyright = f'2022-{datetime.date.today().year}, CNES/CLS'
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
    'jupyter_sphinx',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True

#: Roles linking to the pull requests and issues of the GitHub repository.
extlinks = {
    'issue': ('https://github.com/CNES/zcollection/issues/%s', 'issue #%s'),
    'pr': ('https://github.com/CNES/zcollection/pull/%s', 'PR #%s'),
}

#: The copy button leaves out the shell prompt of the commands.
copybutton_prompt_text = '$ '

#: A cell that raises stops the build.
jupyter_execute_kwargs = {'timeout': -1, 'allow_errors': False}

autodoc_typehints = 'description'
autodoc_type_aliases = dict(
    ArrayLike='ArrayLike',
    DType='DType',
    DTypeLike='DTypeLike',
    Indexer='Indexer',
    NDArray='NDArray',
    NDMaskedArray='NDMaskedArray',
    PartitionCallback='PartitionCallback',
    QueryDict='QueryDict',
    Scalar='Scalar',
)

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
html_theme = 'pydata_sphinx_theme'
html_title = 'ZCollection'
#: The icons of the navigation bar.
ICON_LINKS = [{
    'name': 'GitHub',
    'url': 'https://github.com/CNES/zcollection',
    'icon': 'fa-brands fa-github',
}]

html_theme_options = {
    'logo': {
        'text': 'ZCollection',
    },
    'navigation_depth': 2,
    'icon_links': ICON_LINKS,
}
html_sidebars = {
    'release': [],
}

intersphinx_mapping = {
    'dask': ('https://docs.dask.org/en/latest/', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
    'numcodecs': ('https://numcodecs.readthedocs.io/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable', None),
}

# -- Extension configuration -------------------------------------------------
#: The order of the examples in the gallery.
EXAMPLES = (
    'ex_collection.py',
    'ex_view.py',
    'ex_indexing.py',
    'ex_without_dask.py',
)
EXAMPLE_ORDER = FunctionSortKey(EXAMPLES.index, 'explicit example order')

sphinx_gallery_conf = {
    'examples_dirs': [HERE.parent.parent.joinpath('examples')],
    'filename_pattern': r'[\\\/]ex_',
    'pypandoc': False,
    'within_subsection_order': EXAMPLE_ORDER,
}
