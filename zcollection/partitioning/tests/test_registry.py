# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test the registry module.
=========================
"""
from typing import Any

import pytest

from .. import registry


def test_get_codecs() -> None:
    """Test the get_codecs function."""
    with pytest.raises(ValueError):
        registry.get_codecs({'ID': 'foo'})

    with pytest.raises(ValueError):
        registry.get_codecs({'id': 'foo'})


class MyCodec:
    """A dummy codec."""
    ID = 'foo'

    __slots__ = ('attribute', )

    def __init__(self, attribute) -> None:
        self.attribute: Any = attribute

    def get_config(self) -> dict:
        """Returns the configuration of the codec."""
        return {'id': self.ID, 'attribute': self.attribute}

    @classmethod
    def from_config(cls, config: dict) -> 'MyCodec':
        """Creates an instance from the given configuration."""
        return cls(config['attribute'])


def test_register_codec() -> None:
    """Test the register_codec function."""
    registry.register_codec(MyCodec, codec_id='foo')  # type: ignore[arg-type]

    instance = MyCodec(12)

    other = registry.get_codecs(instance.get_config())
    assert other.attribute == instance.attribute  # type: ignore
    assert isinstance(other, MyCodec)

    with pytest.raises(ValueError):
        registry.register_codec(
            MyCodec,  # type: ignore[arg-type]
            codec_id='foo')
