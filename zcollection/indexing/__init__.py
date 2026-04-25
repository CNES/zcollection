"""Parquet-backed secondary indices over a collection.

An :class:`Indexer` is a row-level lookup table that lets callers find
the ``(partition, row-slice)`` ranges containing a given key value. It
sits beside the collection (typically at ``<root>/_indices/<name>.parquet``)
and is rebuilt by walking partitions through the collection's ``map``.

The v3 indexer is deliberately small compared to the v2 surface: it is a
pyarrow Table on disk and a thin lookup helper. Callers pick the columns
they want indexed via the ``builder`` callable.
"""

from .parquet import IndexBuilder, Indexer


__all__ = ("IndexBuilder", "Indexer")
