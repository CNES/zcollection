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
from .array import Array, new_array
from .delayed_array import DelayedArray, new_delayed_array

__all__ = [
    'Attribute', 'Variable', 'Array', 'DelayedArray', 'new_array',
    'new_delayed_array'
]
