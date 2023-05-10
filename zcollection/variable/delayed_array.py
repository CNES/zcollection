# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Delayed variable arrays.
========================
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Sequence
import uuid

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.base
import dask.highlevelgraph
import dask.threaded
import numcodecs.abc
import numpy
import zarr

from ..meta import Attribute
from ..type_hints import ArrayLike, NDArray, NDMaskedArray
from .abc import Variable, concat, new_variable, not_equal

#: The dask array getter used to access the data.
GETTER: Callable = dask.array.core.getter


def _blockdims_from_blockshape(
        shape: tuple[int, ...],
        chunks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Convert a blockshape to a blockdims tuple."""
    return tuple(((chunk_item, ) * (shape_item // chunk_item) +
                  ((shape_item % chunk_item, ) if shape_item %
                   chunk_item else ()) if shape_item else (0, ))
                 for shape_item, chunk_item in zip(shape, chunks))


def from_zarr_array(
    array: zarr.Array,
    shape: Sequence[int],
    chunks: Sequence[int],
    name: str,
    *,
    lock: bool = False,
    asarray=True,
    inline_array=True,
) -> dask.array.core.Array:
    """Convert a Zarr array to a Dask array with the specified shape and
    chunks.

    Args:
        array: The Zarr array to convert.
        shape: The desired shape of the resulting Dask array.
        chunks: The desired chunk shape of the resulting Dask array.
        name: The name of the resulting Dask array.
        lock: Whether to use a lock to protect the underlying data store.
        asarray: Whether to return the resulting Dask array as an array.
        inline_array: Whether to inline the resulting array data in the tasks.

    Returns:
        A Dask array equivalent to the provided Zarr array.
    """
    dsk: dask.highlevelgraph.HighLevelGraph
    normalized_chunks: tuple[tuple[int, ...], ...]

    normalized_chunks = sum(  # type: ignore[return-value]
        (_blockdims_from_blockshape(
            (shape_item, ),
            (chunk_item, )) if not isinstance(chunk_item, (tuple, list)) else
         (chunk_item, ) for shape_item, chunk_item in zip(shape, chunks)),
        (),
    )
    dsk = dask.array.core.graph_from_arraylike(
        array,
        normalized_chunks,
        shape,
        name,
        getitem=GETTER,
        lock=lock,
        asarray=asarray,
        dtype=array.dtype,
        inline_array=inline_array,
    )
    return dask.array.core.Array(dsk,
                                 name,
                                 normalized_chunks,
                                 meta=array,
                                 dtype=array.dtype)


def _as_dask_array(
    arr: Any,
    *,
    fill_value: Any | None = None,
) -> tuple[dask.array.core.Array, Any]:
    """Convert an array-like object to a dask array.

    Args:
        arr: An array-like object.
        fill_value: The fill value.

    Returns:
        If the data provided is a masked array, the functions return the array
        with masked data replaced by its fill value and the fill value of the
        offered masked array. Otherwise, the provided array and fill value.
    """
    result: dask.array.core.Array = dask.array.core.asarray(arr)
    _meta: Any = result._meta  # pylint: disable=protected-access
    if isinstance(_meta, numpy.ma.MaskedArray):
        if fill_value is not None and not_equal(fill_value, _meta.fill_value):
            raise ValueError(
                f'The fill value {fill_value!r} does not match the fill value '
                f'{_meta.fill_value!r} of the array.')
        return dask.array.ma.filled(result, _meta.fill_value), _meta.fill_value
    return result, fill_value


class ModifiedVariableError(RuntimeError):
    """Raised when a variable has been modified since is was initialized."""

    def __str__(self) -> str:
        """Get the string representation of the exception.

        Returns:
            The string representation of the exception.
        """
        return ('You tried to access the data of a variable that has been '
                'modified since its initialization. Try to re-load the '
                'dataset.')


class DelayedArray(Variable):
    """Access to the chunked data using Dask arrays.

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """

    def __init__(self,
                 name: str,
                 data: ArrayLike[Any],
                 dimensions: Sequence[str],
                 *,
                 attrs: Sequence[Attribute] | None = None,
                 compressor: numcodecs.abc.Codec | None = None,
                 fill_value: Any | None = None,
                 filters: Sequence[numcodecs.abc.Codec] | None = None) -> None:
        array: dask.array.core.Array
        array, fill_value = _as_dask_array(data, fill_value=fill_value)
        # pylint: disable=duplicate-code
        # The code is not duplicated, we need to call the parent constructor,
        # but pylint does not understand that.
        super().__init__(
            name,
            array,
            dimensions,
            attrs=attrs,
            compressor=compressor,
            fill_value=fill_value,
            filters=filters,
        )
        # pylint: enable=duplicate-code

    @property
    def data(self) -> dask.array.core.Array:
        """Return the underlying dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """
        # If the fill value is None, or if the dask array already holds a
        # masked array, return the underlying array.
        # pylint: disable=protected-access
        # No other way to check if the dask array is a masked array.
        if (self.fill_value is None
                or isinstance(self.array._meta, numpy.ma.MaskedArray)):
            return self.array
        # pylint: enable=protected-access
        return dask.array.ma.masked_equal(self.array, self.fill_value)

    @property
    def values(self) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """
        return self.compute()

    @values.setter
    def values(self, data: ArrayLike[Any]) -> None:
        """Defines the underlying dask array. If the data provided is a masked
        array, it's converted to an array, where the masked values are replaced
        by its fill value, and its fill value becomes the new fill value of
        this instance. Otherwise, the underlying array is defined as the new
        data and the fill value is set to None.

        Args:
            data: The new data to use

        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """
        if len(data.shape) != len(self.dimensions):
            raise ValueError('data shape does not match variable dimensions')
        self.array, self.fill_value = _as_dask_array(
            data, fill_value=self.fill_value)

    def persist(self, **kwargs) -> DelayedArray:
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        self.array = dask.base.persist(self.array, **kwargs)
        return self

    def compute(self, **kwargs) -> NDArray | NDMaskedArray:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Args:
            **kwargs: Keyword arguments passed to
                :meth:`dask.array.Array.compute`.
        """
        try:
            values: NDArray
            values, = dask.base.compute(self.array, traverse=False, **kwargs)
        except ValueError as exc:
            msg = str(exc)
            if 'cannot reshape' in msg or 'buffer too small' in msg:
                raise ModifiedVariableError() from exc
            raise
        return values if self.fill_value is None else numpy.ma.masked_equal(
            values, self.fill_value)

    def fill(self) -> DelayedArray:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            self.array = dask.array.creation.full_like(self.array,
                                                       self.fill_value)
        return self

    @classmethod
    def from_zarr(cls, array: zarr.Array, name: str, dimension: str,
                  **kwargs) -> DelayedArray:
        """Create a new variable from a zarr array.

        Args:
            array: The zarr array
            name: Name of the variable
            dimension: Name of the attribute that defines the dimensions of the
                variable
            **kwargs: Keyword arguments passed to
                :func:`dask.array.from_array`

        Returns:
            The variable
        """
        attrs = tuple(
            Attribute(k, v) for k, v in array.attrs.items() if k != dimension)
        data: dask.array.core.Array = from_zarr_array(
            array,
            array.shape,
            array.chunks,
            name=f'{name}-{uuid.uuid1()}',
            **kwargs,
        )
        # pylint: disable=duplicate-code
        # This call is similar to the one in array.py but it's not the same
        # behaviour.
        return new_variable(cls,
                            name=name,
                            array=data,
                            dimensions=array.attrs[dimension],
                            attrs=attrs,
                            compressor=array.compressor,
                            fill_value=array.fill_value,
                            filters=tuple(array.filters or ()))
        # pylint: enable=duplicate-code

    def concat(self, other: DelayedArray | Sequence[DelayedArray],
               dim: str) -> DelayedArray:
        """Concatenate this variable with another variable or a list of
        variables along a dimension.

        Args:
            other: Variable or list of variables to concatenate with this
                variable.
            dim: Dimension to concatenate along.

        Returns:
            New variable.

        Raises:
            ValueError: if the variables provided is an empty sequence.
        """
        return concat(self, other, dask.array.core.concatenate, dim)

    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.
        Returns:
            The variable slice.
        """
        return self.data[key]

    def rechunk(self, **kwargs) -> DelayedArray:
        """Rechunk the variable.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.rechunk`

        Returns:
            The variable.
        """
        # pylint: disable=duplicate-code
        # False positive with the method concat.
        return new_variable(type(self),
                            name=self.name,
                            array=self.array.rechunk(**kwargs),
                            dimensions=self.dimensions,
                            attrs=self.attrs,
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)
        # pylint: enable=duplicate-code

    def to_dask_array(self):
        """Return the underlying dask array.

        Returns:
            The underlying dask array

        .. seealso::

            :func:`dask.array.asarray`
        """
        return self.data

    def __dask_graph__(self) -> Mapping | None:
        """Return the dask Graph."""
        return self.data.__dask_graph__()

    def __dask_keys__(self) -> list:
        """Return the output keys for the Dask graph."""
        return self.data.__dask_keys__()

    def __dask_layers__(self) -> tuple:
        """Return the layers for the Dask graph."""
        return self.data.__dask_layers__()

    def __dask_tokenize__(self) -> Any:
        """Return the token for the Dask graph."""
        return dask.base.normalize_token(
            (type(self), self.name, self.data, self.dimensions, self.attrs,
             self.fill_value))

    @staticmethod
    def __dask_optimize__(dsk: MutableMapping, keys: list,
                          **kwargs) -> MutableMapping:
        """Returns whether the Dask graph can be optimized.

        .. seealso::
            :meth:`dask.array.Array.__dask_optimize__`
        """
        return dask.array.core.Array.__dask_optimize__(dsk, keys, **kwargs)

    #: The default scheduler get to use for this object.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def _dask_finalize(self, results, array_func, *args,
                       **kwargs) -> DelayedArray:
        """Finalize the computation of the variable."""
        array: Any = array_func(results, *args, **kwargs)
        if not isinstance(array, dask.array.core.Array):
            array = dask.array.core.from_array(array)
        # pylint: disable=duplicate-code
        # False positive with the method metadata defined in the base class.
        return new_variable(type(self),
                            name=self.name,
                            array=array,
                            dimensions=self.dimensions,
                            attrs=self.attrs,
                            compressor=self.compressor,
                            fill_value=self.fill_value,
                            filters=self.filters)
        # pylint: enable=duplicate-code

    def __dask_postcompute__(self) -> tuple:
        """Return the finalizer and extra arguments to convert the computed
        results into their in-memory representation."""
        array_func: Callable
        array_args: tuple

        array_func, array_args = self.data.__dask_postcompute__()
        return self._dask_finalize, (array_func, ) + array_args

    def __dask_postpersist__(self) -> tuple:
        """Return the rebuilder and extra arguments to rebuild an equivalent
        Dask collection from a persisted or rebuilt graph."""
        array_func: Callable
        array_args: tuple

        array_func, array_args = self.data.__dask_postpersist__()
        return self._dask_finalize, (array_func, ) + array_args
