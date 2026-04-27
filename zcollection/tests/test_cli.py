# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""CLI sanity tests — drives ``ls / inspect / drop`` via :func:`cli.main`."""

import io
import json

import pytest

import zcollection as zc
from zcollection.cli.main import main


@pytest.fixture
def populated(tmp_path, schema, dataset, partitioning):
    """Provide the path to a populated local collection for CLI tests."""
    path = str(tmp_path / "col")
    store = zc.LocalStore(path)
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)
    return path


def _run(argv, stdin: str = "") -> tuple[int, str]:
    out = io.StringIO()
    rc = main(argv, out=out, stdin=io.StringIO(stdin))
    return rc, out.getvalue()


def test_ls_lists_partitions(populated):
    """``ls`` prints all partitions of the collection."""
    rc, txt = _run(["ls", populated])
    assert rc == 0
    assert sorted(txt.split()) == ["num=0", "num=1", "num=2"]


def test_ls_filter(populated):
    """``ls --filter`` narrows the partition listing."""
    rc, txt = _run(["ls", populated, "--filter", "num == 1"])
    assert rc == 0
    assert txt.strip().splitlines() == ["num=1"]


def test_ls_json(populated):
    """``ls --json`` emits the partition list as a JSON array."""
    rc, txt = _run(["ls", populated, "--json"])
    assert rc == 0
    assert sorted(json.loads(txt)) == ["num=0", "num=1", "num=2"]


def test_inspect_text(populated):
    """``inspect`` prints axis, partition count, and variable summary in text."""
    rc, txt = _run(["inspect", populated])
    assert rc == 0
    assert "axis:         num" in txt
    assert "partitions:   3" in txt
    assert "value:" in txt
    assert "static:" in txt
    assert "(immutable)" in txt


def test_inspect_json(populated):
    """``inspect --json`` emits structured collection metadata."""
    rc, txt = _run(["inspect", populated, "--json"])
    assert rc == 0
    info = json.loads(txt)
    assert info["axis"] == "num"
    assert info["n_partitions"] == 3
    assert info["variables"]["static"]["immutable"] is True


def test_drop_with_yes_flag(populated):
    """``drop -y`` drops matching partitions without prompting."""
    rc, txt = _run(["drop", populated, "--filter", "num == 1", "-y"])
    assert rc == 0
    assert "dropped 1 partition" in txt
    _, ls = _run(["ls", populated])
    assert sorted(ls.split()) == ["num=0", "num=2"]


def test_drop_aborts_without_confirmation(populated):
    """``drop`` aborts when stdin lacks a positive confirmation."""
    rc, txt = _run(["drop", populated, "--filter", "num == 1"], stdin="\n")
    assert rc == 1
    assert "aborted" in txt
    _, ls = _run(["ls", populated])
    assert sorted(ls.split()) == ["num=0", "num=1", "num=2"]


def test_drop_confirmed_via_stdin(populated):
    """``drop`` proceeds when stdin sends ``y``."""
    rc, _ = _run(
        ["drop", populated, "--filter", "num == 0"],
        stdin="y\n",
    )
    assert rc == 0
    _, ls = _run(["ls", populated])
    assert sorted(ls.split()) == ["num=1", "num=2"]


def test_drop_no_match(populated):
    """``drop`` reports cleanly when the filter matches no partitions."""
    rc, txt = _run(["drop", populated, "--filter", "num == 99", "-y"])
    assert rc == 0
    assert "no partitions match" in txt


def test_missing_collection_returns_error(tmp_path):
    """Inspecting a missing collection exits with code 2."""
    rc, _ = _run(["inspect", str(tmp_path / "nope")])
    assert rc == 2
