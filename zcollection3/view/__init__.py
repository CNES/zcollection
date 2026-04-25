"""Views — additional variables overlaid onto a base collection.

A :class:`View` is an *overlay* persisted in its own store. It declares
extra variables that share the base collection's partitioning. Each base
partition has a sibling group in the view store holding only the view's
own variables; reads merge view variables on top of the base partition.

Layout::

    <view_root>/_zcollection_view.json   # config: base URI, var schemas
    <view_root>/<partition>/zarr.json    # only view-owned arrays

Restrictions for the v3 port:

- No overlap support (the v2 ``select_overlap`` helper is not ported).
- No checksum/repair logic — partition presence is the source of truth.
- Views are read-write only against a writable base collection handle;
  the view does *not* mutate the base.
"""
from __future__ import annotations

from .base import View, ViewReference

__all__ = ("View", "ViewReference")
