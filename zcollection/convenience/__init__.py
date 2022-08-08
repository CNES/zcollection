# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from .collection import create_collection, open_collection
from .view import create_view, open_view

__all__ = ['create_collection', 'open_collection', 'create_view', 'open_view']
