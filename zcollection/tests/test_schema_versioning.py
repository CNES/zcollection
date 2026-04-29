# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the on-disk format versioning helpers."""

from collections.abc import Iterator

import pytest

from zcollection.errors import FormatVersionError
from zcollection.schema import versioning


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot and restore the module-level upgrader registry."""
    saved = dict(versioning._REGISTRY)
    versioning._REGISTRY.clear()
    try:
        yield
    finally:
        versioning._REGISTRY.clear()
        versioning._REGISTRY.update(saved)


def test_upgrade_noop_when_at_current_version():
    """``upgrade`` returns the payload unchanged when already current."""
    payload = {"format_version": versioning.FORMAT_VERSION, "extra": 1}
    assert versioning.upgrade(payload) == payload


def test_upgrade_defaults_to_version_one_when_missing():
    """Missing ``format_version`` is treated as 1 (current), so no error."""
    # FORMAT_VERSION is 1; payload without the key should pass through.
    payload = {"schema": {"variables": []}}
    assert versioning.upgrade(payload) == payload


def test_upgrade_runs_chain_in_order(
    clean_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered chain ``1 → 2 → 3`` runs upgraders in order."""
    monkeypatch.setattr(versioning, "FORMAT_VERSION", 3)

    @versioning.register(1, 2)
    def _v1_to_v2(p: dict) -> dict:
        return {**p, "format_version": 2, "steps": [*p.get("steps", []), "v2"]}

    @versioning.register(2, 3)
    def _v2_to_v3(p: dict) -> dict:
        return {**p, "format_version": 3, "steps": [*p["steps"], "v3"]}

    out = versioning.upgrade({"format_version": 1})
    assert out["format_version"] == 3
    assert out["steps"] == ["v2", "v3"]


def test_upgrade_missing_intermediate_raises(
    clean_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``FormatVersionError`` is raised if the chain has a gap."""
    monkeypatch.setattr(versioning, "FORMAT_VERSION", 3)

    # Only register 1 -> 2; 2 -> 3 is missing.
    @versioning.register(1, 2)
    def _v1_to_v2(p: dict) -> dict:
        return {**p, "format_version": 2}

    with pytest.raises(FormatVersionError, match="no upgrader registered"):
        versioning.upgrade({"format_version": 1})


def test_upgrade_future_version_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload newer than ``FORMAT_VERSION`` raises ``FormatVersionError``."""
    monkeypatch.setattr(versioning, "FORMAT_VERSION", 1)
    with pytest.raises(FormatVersionError, match="newer than supported"):
        versioning.upgrade({"format_version": 99})


def test_register_returns_the_decorated_callable(clean_registry: None) -> None:
    """``register`` is a decorator that returns the original function."""

    def upgrader(p: dict) -> dict:
        return p

    result = versioning.register(1, 2)(upgrader)
    assert result is upgrader
    assert versioning._REGISTRY[(1, 2)] is upgrader
