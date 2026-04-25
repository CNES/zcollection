"""Collection-level flows."""

from . import merge
from .base import Collection
from .merge import MergeCallable


__all__ = ("Collection", "MergeCallable", "merge")
