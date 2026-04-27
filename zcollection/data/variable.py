# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Single, polymorphic Variable bound to a ``VariableSchema``.

A :class:`Variable` is an array plus its schema. The array can be a
:class:`numpy.ndarray` (eager) or any non-numpy array-like —
``dask.array.Array``, a Zarr ``AsyncArray`` proxy, or a custom lazy
backend. The :attr:`Variable.is_lazy` flag is true iff the underlying
array isn't a plain :class:`numpy.ndarray`.
"""

from typing import TYPE_CHECKING, Any
import math

import numpy


if TYPE_CHECKING:
    from ..schema import VariableSchema


class Variable:
    """A named array bound to a :class:`VariableSchema`.

    ``data`` may be a :class:`numpy.ndarray` (eager), any object exposing
    a ``compute()`` method (dask-style lazy arrays), an arbitrary
    array-like (anything :func:`numpy.asarray` accepts), or ``None``
    (declared but not yet populated). :attr:`is_lazy` is true iff the
    array isn't a plain :class:`numpy.ndarray`.

    On construction the data is validated against the schema's number
    of dimensions; dtype mismatches are *not* enforced (upcasts are
    accepted silently).

    Args:
        schema: Variable schema describing dtype, dims and metadata.
        data: Underlying array, or ``None`` for a placeholder variable.

    Raises:
        ValueError: If ``data`` exposes an ``ndim`` attribute that
            disagrees with ``schema.ndim``.

    """

    __slots__ = ("_data", "schema")

    def __init__(self, schema: VariableSchema, data: Any) -> None:
        """Initialize a Variable."""
        #: The variable schema describing dtype, dims and metadata.
        self.schema = schema
        # Underlying array (numpy / dask / array-like / None).
        self._data = data
        # Validate the data's ndim against the schema.
        self._validate()

    def _validate(self) -> None:
        """Reject data whose dim count doesn't match :attr:`schema`."""
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
        """Return the shape of the underlying data.

        Returns ``()`` when :attr:`data` is ``None`` or has no ``shape``
        attribute (the empty tuple makes the variable look like a 0-D
        scalar to size-aware consumers).
        """
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
        """Return a fresh copy of the schema attributes.

        The returned dict is detached: mutating it does not affect the
        underlying schema.
        """
        return dict(self.schema.attrs)

    @property
    def is_lazy(self) -> bool:
        """Return whether the underlying data isn't a plain ``numpy.ndarray``.

        True for dask arrays, Zarr ``AsyncArray`` proxies, and anything
        else that isn't a concrete in-memory numpy buffer; false when
        the data is already materialised. ``data is None`` also returns
        true (the placeholder is treated as not-yet-eager).
        """
        return not isinstance(self._data, numpy.ndarray)

    @property
    def data(self) -> Any:
        """Return the underlying array as-is (no materialisation)."""
        return self._data

    @property
    def nbytes(self) -> int:
        """Return the uncompressed byte size of the underlying data.

        Computed as ``prod(shape) * dtype.itemsize`` — the same
        convention as :attr:`numpy.ndarray.nbytes`. Ignores any
        compression or sharding the variable might carry on disk.
        Returns ``0`` for placeholder variables (``data is None``).
        """
        return (
            0
            if self._data is None
            else math.prod(self.shape) * self.dtype.itemsize
        )

    def to_numpy(self) -> numpy.ndarray:
        """Materialise the data as a numpy array.

        Dispatches in three cases:

        - already a :class:`numpy.ndarray` → returned as-is (no copy).
        - has a ``compute()`` method (dask-style lazy arrays) → call it
          and return the materialised result.
        - otherwise → :func:`numpy.asarray` on the data.

        Calling this on a Variable with ``data is None`` produces
        ``numpy.array(None, dtype=object)``, which is rarely useful;
        guard against that case at the caller.
        """
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
