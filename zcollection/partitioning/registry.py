# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Registers the partitioning codecs.
==================================
"""
from typing import Dict

from . import abc

#: A registry of all available partitioning codecs.
CODEC_REGISTRY: Dict[str, abc.Partitioning] = {}


def get_codecs(config: Dict) -> abc.Partitioning:
    """Get the partitioning scheme for the given configuration.

    Args:
        config: A dictionary of the partitioning configuration parameters.

    Returns:
        The partitioning scheme.

    Raises:
        ValueError: If the requested codec is not defined.
    """
    codec_id = config.pop('id', None)
    if codec_id is None:
        raise ValueError(f'codec not available: {codec_id!r}')
    cls = CODEC_REGISTRY.get(codec_id, None)
    if cls is None:
        raise ValueError(f'codec not available: {codec_id!r}')
    return cls.from_config(config)


def register_codec(cls, codec_id=None) -> None:
    """Register a partitioning scheme.

    Args:
        cls: The partitioning scheme class.
        codec_id: The partitioning scheme identifier.

    Raises:
        ValueError: If the codec identifier is already registered.
    """
    if codec_id is None:
        codec_id = cls.ID
    if codec_id in CODEC_REGISTRY:
        raise ValueError(f'codec already registered: {codec_id!r}')
    CODEC_REGISTRY[codec_id] = cls
