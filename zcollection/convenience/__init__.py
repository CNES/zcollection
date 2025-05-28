# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from .collection import (
    create_collection,
    open_collection,
    update_deprecated_collection,
)
from .view import create_view, open_view, update_deprecated_view

__all__ = (
    'create_collection',
    'create_view',
    'open_collection',
    'open_view',
    'update_deprecated_collection',
    'update_deprecated_view',
)
