"""Regression: insert+query must not emit any ``ZarrUserWarning``.

We deliberately do not call :func:`zarr.consolidate_metadata` (the v3 spec
hasn't blessed the ``consolidated_metadata`` field). zcollection's external
schema already provides everything consolidation would have cached, so we
rely on zarr's native per-array ``zarr.json`` reads. This test guards
against any future regression that re-introduces the warning.
"""

import warnings

import zarr.errors

import zcollection as zc


def test_insert_then_query_emits_no_zarr_user_warning(
    tmp_path,
    schema,
    dataset,
    partitioning,
):
    """Insert and query must not raise any ``ZarrUserWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", zarr.errors.ZarrUserWarning)
        store = zc.LocalStore(tmp_path / "col")
        col = zc.create_collection(
            store,
            schema=schema,
            axis="num",
            partitioning=partitioning,
            overwrite=True,
        )
        col.insert(dataset)
        out = zc.open_collection(store, mode="r").query()
        assert out is not None
