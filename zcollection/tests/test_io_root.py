# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Round-trip tests for the root-config I/O helpers."""

import pytest

import zcollection as zc
from zcollection.errors import CollectionNotFoundError
from zcollection.io.root import read_root_config, write_root_config
from zcollection.schema.serde import CONFIG_FILE


def test_write_then_read_roundtrips(schema, partitioning):
    """Round-trip through ``write_root_config`` / ``read_root_config``."""
    store = zc.MemoryStore()
    write_root_config(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning.to_json(),
        catalog_enabled=False,
    )

    doc = read_root_config(store)
    assert doc["axis"] == "num"
    assert doc["catalog"] == {"enabled": False}
    assert doc["partitioning"] == partitioning.to_json()
    assert "schema" in doc
    assert "format_version" in doc


def test_write_includes_extras(schema, partitioning):
    """``extras`` is propagated into the persisted document."""
    store = zc.MemoryStore()
    write_root_config(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning.to_json(),
        catalog_enabled=True,
        extras={"created_by": "test", "build": 7},
    )

    doc = read_root_config(store)
    assert doc["catalog"] == {"enabled": True}
    assert doc["extras"] == {"created_by": "test", "build": 7}


def test_top_level_keys_are_stable(schema, partitioning):
    """Snapshot the document's top-level shape — accidental changes break this."""
    store = zc.MemoryStore()
    write_root_config(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning.to_json(),
        catalog_enabled=False,
    )
    doc = read_root_config(store)
    assert set(doc.keys()) == {
        "format_version",
        "axis",
        "partitioning",
        "catalog",
        "schema",
    }


def test_read_missing_config_raises_not_found():
    """``read_root_config`` raises ``CollectionNotFoundError`` when absent."""
    store = zc.MemoryStore()
    with pytest.raises(CollectionNotFoundError):
        read_root_config(store)


def test_read_corrupt_config_raises_value_error():
    """A non-JSON payload at the config path surfaces as a parse error."""
    store = zc.MemoryStore()
    store.write_bytes(CONFIG_FILE, b"{ this is not json")
    with pytest.raises(ValueError, match=r"(?i)json|expecting"):
        read_root_config(store)
