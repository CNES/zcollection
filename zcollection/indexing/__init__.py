# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Indexing a Collection.
======================
"""
import warnings

try:
    from .abc import Indexer, QueryDict, Scalar
    __all__ = ['Indexer', 'QueryDict', 'Scalar']
except ImportError:  # pragma: no cover
    warnings.warn(
        'Install PyArrow to use the indexing capabilities of zcollection.')
