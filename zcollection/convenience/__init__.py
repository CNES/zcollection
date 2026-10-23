# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from .collection import (
    check,
    create_collection,
    migrate,
    open_collection,
    prune,
    status,
)
from .view import (
    check_view,
    create_view,
    migrate_view,
    open_view,
    prune_view,
    view_status,
)

__all__ = (
    'check',
    'check_view',
    'create_collection',
    'create_view',
    'migrate',
    'migrate_view',
    'open_collection',
    'open_view',
    'prune',
    'prune_view',
    'status',
    'view_status',
)
