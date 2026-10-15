# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Variables of a dataset.
=======================
"""
from ..meta import Attribute
from .abc import Variable
from .array import Array
from .delayed_array import DelayedArray

__all__ = ('Array', 'Attribute', 'DelayedArray', 'Variable')
