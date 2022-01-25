# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning scheme.
====================
"""
from .abc import Partitioning
from .date import Date
from .expression import Expression
from .registry import get_codecs, register_codec
from .sequence import Sequence

register_codec(Date)
register_codec(Sequence)

__all__ = ["Partitioning", "Date", "Expression", "Sequence", "get_codecs"]
