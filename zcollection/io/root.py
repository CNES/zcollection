"""Read/write the ``_zcollection.json`` root config."""

from typing import TYPE_CHECKING, Any

from ..errors import CollectionNotFoundError
from ..schema.serde import CONFIG_FILE, decode_root, encode_root

if TYPE_CHECKING:
    from ..schema import DatasetSchema
    from ..store import Store


def write_root_config(
    store: Store,
    *,
    schema: DatasetSchema,
    axis: str,
    partitioning: dict[str, Any],
    catalog_enabled: bool,
    extras: dict[str, Any] | None = None,
) -> None:
    """Encode and write the root collection config to ``store``."""
    payload = encode_root(
        schema=schema,
        axis=axis,
        partitioning=partitioning,
        catalog_enabled=catalog_enabled,
        extras=extras,
    )
    store.write_bytes(CONFIG_FILE, payload)


def read_root_config(store: Store) -> dict[str, Any]:
    """Read and decode the root collection config from ``store``."""
    raw = store.read_bytes(CONFIG_FILE)
    if raw is None:
        raise CollectionNotFoundError(f"no {CONFIG_FILE} at {store.root_uri}")
    return decode_root(raw)
