"""Single, polymorphic Variable: holds numpy or dask data plus its schema."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy


if TYPE_CHECKING:
    from ..schema import VariableSchema


class Variable:
    """A named array bound to a :class:`VariableSchema`.

    ``data`` is either a :class:`numpy.ndarray` (eager) or a
    :class:`dask.array.Array` (lazy). The dataset is "lazy" iff any of its
    variables holds a non-numpy backend.
    """

    __slots__ = ("_data", "schema")

    def __init__(self, schema: VariableSchema, data: Any) -> None:
        self.schema = schema
        self._data = data
        self._validate()

    def _validate(self) -> None:
        if self._data is None:
            return
        ndim = getattr(self._data, "ndim", None)
        if ndim is not None and ndim != self.schema.ndim:
            raise ValueError(
                f"variable {self.schema.name!r}: data has {ndim} dims, "
                f"schema declares {self.schema.ndim}"
            )
        # dtype check: allow upcasting silently for now; strict mode optional.
        return None

    # Public, frozen-ish accessors -----------------------------------

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def dimensions(self) -> tuple[str, ...]:
        return self.schema.dimensions

    @property
    def dtype(self) -> numpy.dtype:
        return self.schema.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self._data, "shape", ()))

    @property
    def ndim(self) -> int:
        return self.schema.ndim

    @property
    def fill_value(self) -> Any:
        return self.schema.fill_value

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self.schema.attrs)

    @property
    def is_lazy(self) -> bool:
        return not isinstance(self._data, numpy.ndarray)

    @property
    def data(self) -> Any:
        return self._data

    def to_numpy(self) -> numpy.ndarray:
        d = self._data
        if isinstance(d, numpy.ndarray):
            return d
        if hasattr(d, "compute"):
            return d.compute()
        return numpy.asarray(d)

    def __repr__(self) -> str:
        return (
            f"Variable(name={self.name!r}, dims={self.dimensions}, "
            f"dtype={self.dtype}, shape={self.shape}, lazy={self.is_lazy})"
        )
