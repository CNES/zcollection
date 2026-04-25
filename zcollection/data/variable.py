"""Single, polymorphic Variable: holds numpy or dask data plus its schema."""

from typing import TYPE_CHECKING, Any
import math

import numpy


if TYPE_CHECKING:
    from ..schema import VariableSchema


class Variable:
    """A named array bound to a :class:`VariableSchema`.

    ``data`` is either a :class:`numpy.ndarray` (eager) or a
    :class:`dask.array.Array` (lazy). The dataset is "lazy" iff any of its
    variables holds a non-numpy backend.

    Args:
        schema: Variable schema describing dtype, dims and metadata.
        data: Underlying array (numpy or dask).

    """

    __slots__ = ("_data", "schema")

    def __init__(self, schema: VariableSchema, data: Any) -> None:
        """Initialize a Variable."""
        #: The variable schema describing dtype, dims and metadata.
        self.schema = schema
        #: The underlying array (numpy or dask).
        self._data = data
        #: Validate the data against the schema (e.g. check number of
        #: dimensions).
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
        """Return the variable name."""
        return self.schema.name

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Return the dimension names."""
        return self.schema.dimensions

    @property
    def dtype(self) -> numpy.dtype:
        """Return the numpy dtype."""
        return self.schema.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data."""
        return tuple(getattr(self._data, "shape", ()))

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.schema.ndim

    @property
    def fill_value(self) -> Any:
        """Return the schema fill value."""
        return self.schema.fill_value

    @property
    def attrs(self) -> dict[str, Any]:
        """Return a copy of the schema attributes."""
        return dict(self.schema.attrs)

    @property
    def is_lazy(self) -> bool:
        """Return whether the underlying data is non-numpy (e.g. dask)."""
        return not isinstance(self._data, numpy.ndarray)

    @property
    def data(self) -> Any:
        """Return the underlying array as-is."""
        return self._data

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes spanned by the underlying data."""
        return math.prod(self.shape) * self.dtype.itemsize

    def to_numpy(self) -> numpy.ndarray:
        """Materialise the data as a numpy array."""
        d = self._data
        if isinstance(d, numpy.ndarray):
            return d
        if hasattr(d, "compute"):
            return d.compute()
        return numpy.asarray(d)

    def __repr__(self) -> str:
        """Return a multi-line, xarray-like representation of the variable."""
        from ._repr import variable_repr

        return variable_repr(self)
