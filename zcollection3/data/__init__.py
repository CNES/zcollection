"""In-memory data containers (single :class:`Variable`, :class:`Dataset`)."""
from __future__ import annotations

from .dataset import Dataset
from .variable import Variable

__all__ = ("Dataset", "Variable")
