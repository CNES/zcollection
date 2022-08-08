# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test the registry module.
=========================
"""
import pytest

#
from .. import registry


def test_get_codecs():
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
        self.attribute = attribute

    def get_config(self) -> dict:
        """Returns the configuration of the codec."""
        return {'id': self.ID, 'attribute': self.attribute}

    @classmethod
    def from_config(cls, config: dict) -> 'MyCodec':
        """Creates an instance from the given configuration."""
        return cls(config['attribute'])


def test_register_codec():
    """Test the register_codec function."""
    registry.register_codec(MyCodec, 'foo')

    instance = MyCodec(12)

    other = registry.get_codecs(instance.get_config())
    assert other.attribute == instance.attribute  # type: ignore
    assert isinstance(other, MyCodec)

    with pytest.raises(ValueError):
        registry.register_codec(MyCodec, 'foo')
