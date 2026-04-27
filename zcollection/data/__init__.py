# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""In-memory data containers.

Three classes work together:

- :class:`Variable` — a named array bound to a
  :class:`~zcollection.schema.VariableSchema`. Holds a
  :class:`numpy.ndarray` (eager) or any non-numpy array-like
  (``dask.array.Array`` and other lazy backends).
- :class:`Group` — a named container of variables, attributes,
  dimensions, and child groups. Mirrors a Zarr v3 group on disk and
  supports path-based lookups, dimension inheritance, and
  whole-tree byte accounting.
- :class:`Dataset` — the root :class:`Group` bound to a
  :class:`~zcollection.schema.DatasetSchema`. Adds the xarray
  bridge (:meth:`Dataset.to_xarray`, :meth:`Dataset.from_xarray`)
  and a path-aware constructor that routes variables into nested
  groups on the way in.
"""

from .dataset import Dataset
from .group import Group
from .variable import Variable


__all__ = ("Dataset", "Group", "Variable")
