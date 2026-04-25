"""Date partitioning — bucket a datetime64 axis by Y/M/D/h/m/s."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import numpy

from ..errors import PartitionError
from .base import PartitionKey, runs_from_inverse

if TYPE_CHECKING:
    from ..data import Dataset

#: Mapping from a resolution code to the (component-name, datetime64 unit, padding) tuple.
_RESOLUTIONS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "Y": (("year", "Y", 4),),
    "M": (("year", "Y", 4), ("month", "M", 2)),
    "D": (("year", "Y", 4), ("month", "M", 2), ("day", "D", 2)),
    "h": (
        ("year", "Y", 4),
        ("month", "M", 2),
        ("day", "D", 2),
        ("hour", "h", 2),
    ),
    "m": (
        ("year", "Y", 4),
        ("month", "M", 2),
        ("day", "D", 2),
        ("hour", "h", 2),
        ("minute", "m", 2),
    ),
    "s": (
        ("year", "Y", 4),
        ("month", "M", 2),
        ("day", "D", 2),
        ("hour", "h", 2),
        ("minute", "m", 2),
        ("second", "s", 2),
    ),
}


class Date:
    """Partition by truncating a 1-D datetime64 variable to ``resolution``.

    Component names match the v2 layout (``year=2024/month=03/day=01``).
    """

    name = "date"

    def __init__(
        self,
        variables: tuple[str, ...] | str,
        *,
        resolution: str,
        dimension: str | None = None,
    ) -> None:
        if isinstance(variables, str):
            variables = (variables,)
        if len(variables) != 1:
            raise PartitionError(
                f"Date takes exactly one variable; got {variables!r}"
            )
        if resolution not in _RESOLUTIONS:
            raise PartitionError(
                f"unsupported resolution {resolution!r}; "
                f"choose from {tuple(_RESOLUTIONS)!r}"
            )
        self._variable = variables[0]
        self._resolution = resolution
        self._dimension = dimension or variables[0]
        self._components = _RESOLUTIONS[resolution]

    @property
    def axis(self) -> tuple[str, ...]:
        return tuple(name for name, _unit, _pad in self._components)

    @property
    def dimension(self) -> str:
        return self._dimension

    @property
    def resolution(self) -> str:
        return self._resolution

    def split(self, dataset: Dataset) -> Iterator[tuple[PartitionKey, slice]]:
        if self._variable not in dataset:
            raise PartitionError(
                f"variable {self._variable!r} required for Date partitioning is missing"
            )
        var = dataset[self._variable]
        if var.dimensions != (self._dimension,):
            raise PartitionError(
                f"Date variable {self._variable!r} must be 1-D along "
                f"{self._dimension!r}; got dims={var.dimensions}"
            )

        values = var.to_numpy()
        if not numpy.issubdtype(values.dtype, numpy.datetime64):
            raise PartitionError(
                f"Date partitioning requires datetime64 values; "
                f"got dtype {values.dtype}"
            )

        # Truncate to the partition resolution.
        last_unit = self._components[-1][1]
        bucketed = values.astype(f"datetime64[{last_unit}]")
        unique, inverse = numpy.unique(bucketed, return_inverse=True)

        for gid, sl in runs_from_inverse(inverse):
            key = self._key_from_datetime(unique[gid])
            yield key, sl

    def _key_from_datetime(self, value: numpy.datetime64) -> PartitionKey:
        # Use Python datetime conversion for portability.
        ts = value.astype("datetime64[s]").item()
        parts: list[tuple[str, int]] = []
        getters = {
            "year": ts.year,
            "month": ts.month,
            "day": ts.day,
            "hour": ts.hour,
            "minute": ts.minute,
            "second": ts.second,
        }
        for name, _unit, _pad in self._components:
            parts.append((name, int(getters[name])))
        return tuple(parts)

    def encode(self, key: PartitionKey) -> str:
        pad_by_name = {name: pad for name, _unit, pad in self._components}
        return "/".join(
            f"{name}={int(val):0{pad_by_name[name]}d}" for name, val in key
        )

    def decode(self, path: str) -> PartitionKey:
        parts: list[tuple[str, int]] = []
        for token in path.strip("/").split("/"):
            if "=" not in token:
                raise PartitionError(
                    f"invalid partition path segment: {token!r}"
                )
            name, raw = token.split("=", 1)
            parts.append((name, int(raw)))
        return tuple(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variable": self._variable,
            "resolution": self._resolution,
            "dimension": self._dimension,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Date:
        return cls(
            (payload["variable"],),
            resolution=payload["resolution"],
            dimension=payload.get("dimension"),
        )
