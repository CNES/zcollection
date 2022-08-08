# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning scheme.
====================

Entry point of the implemented partitioning schemes.

* :py:class:`Sequence <zcollection.partitioning.sequence.Sequence>`:
  Partitioning a sequence of variables.
* :py:class:`Date <zcollection.partitioning.date.Date>`: Partitioning a
  sequence of dates.

.. class:: Partitioning

    Alias for :class:`zcollection.partitioning.abc.Partitioning`.
"""
from .abc import Partitioning
from .date import Date
from .registry import get_codecs, register_codec
from .sequence import Sequence

register_codec(Date)
register_codec(Sequence)

__all__ = ['Partitioning', 'Date', 'Sequence', 'get_codecs']
