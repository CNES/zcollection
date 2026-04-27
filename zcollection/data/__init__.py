# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""In-memory data containers (:class:`Variable`, :class:`Group`, :class:`Dataset`)."""

from .dataset import Dataset
from .group import Group
from .variable import Variable


__all__ = ("Dataset", "Group", "Variable")
