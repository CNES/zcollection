"""Collection-level flows."""
from __future__ import annotations

from . import merge
from .base import Collection
from .merge import MergeCallable

__all__ = ("Collection", "MergeCallable", "merge")
