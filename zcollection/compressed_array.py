# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Compressed arrays
=================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.array.backends
import dask.array.core
import dask.array.dispatch
import dask.base
import numpy
import numpy.lib.mixins
import zarr

from .type_hints import DType, NDArray, NDMaskedArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from zarr.core.buffer import NDArrayLikeOrScalar

    from .codecs import Codec

#: Type of arrays returned when a compressed array is decompressed.
Array = NDArray | NDMaskedArray


class CompressedArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    """Provide a NumPy-compatible interface to compressed array data.

    The array is compressed using Zarr. Operations decompress its values,
    perform the requested operation, and return a NumPy array. Rechunking
    preserves the source array's codecs.

    Args:
        array: A NumPy array, Zarr array, or CompressedArray.
        **kwargs: Arguments passed to ``zarr.create_array`` when creating the
            in-memory array.
    """

    def __init__(
        self,
        array: CompressedArray | NDArray | zarr.Array,
        **kwargs,
    ):
        self._fill_value = kwargs.get('fill_value', None)
        if isinstance(array, numpy.ndarray):
            self._array = zarr.create_array(store={}, data=array, **kwargs)
        elif isinstance(array, zarr.Array):
            if kwargs:
                raise ValueError('kwargs are not allowed when array is a '
                                 'zarr.Array')
            self._array = array
        elif isinstance(array, CompressedArray):
            self._array = array._array
            self._fill_value = array._fill_value
        else:
            raise TypeError('array must be a numpy.ndarray or a zarr.Array')

    __array_priority__ = 0

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} ({self.shape}) {self.dtype}>'

    def __getitem__(
        self,
        key: int | tuple[int | slice, ...],
    ) -> NDArrayLikeOrScalar | NDMaskedArray:
        """Return the values selected by an index or slice.

        Args:
            key: The index or slice to retrieve.

        Returns:
            The selected values.
        """
        values: NDArrayLikeOrScalar | NDMaskedArray = self._array[key]
        if self._fill_value is not None:
            values = numpy.ma.masked_equal(values, self._fill_value)
        return values

    def __setitem__(
        self,
        key: int | tuple[int | slice, ...],
        value: Array,
    ) -> None:
        """Assign values to an index or slice of the array.

        Args:
            key: The index or slice to update.
            value: The values to assign.
        """
        self._array[key] = value

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._array.ndim

    @property
    def nchunks(self) -> int:
        """Number of chunks."""
        return self._array.nchunks

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self._array.shape

    @property
    def size(self) -> int:
        """The total number of elements in the array."""
        return self._array.size

    @property
    def dtype(self) -> DType:
        """The data type of the array."""
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        """The total number of bytes used by the array."""
        return self._array.nbytes

    @property
    def compressors(self) -> tuple[Codec, ...]:
        """The codecs used to compress the array chunks."""
        return self._array.compressors

    @property
    def fill_value(self) -> Any:
        """The value used for uninitialized portions of the array."""
        return self._fill_value

    @property
    def filters(self) -> tuple[Codec, ...]:
        """The codecs applied to the values before compression."""
        return self._array.filters

    def reshape(self, *shape: int) -> Array:
        """Return a new array containing the same data with a different shape.

        Args:
            *shape: The new shape.

        Returns:
            The reshaped array.
        """
        return self.__array__().reshape(*shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        """The chunk size along each dimension."""
        return self._array.chunks

    def rechunk(self, chunks: tuple[int, ...]) -> CompressedArray:
        """Return the array with a different chunk layout.

        Args:
            chunks: The new chunk sizes.

        Returns:
            The rechunked array.
        """
        return CompressedArray(numpy.asanyarray(self._array[...],
                                                dtype=self.dtype),
                               chunks=chunks,
                               compressors=self._array.compressors,
                               filters=self._array.filters,
                               serializer=self._array.serializer,
                               fill_value=self._fill_value)

    def __array__(self, dtype=None, copy: bool | None = None) -> Array:
        """Return the array as a NumPy array.

        Args:
            dtype: The requested data type.
            copy: Whether to copy the data. Decompression always creates a
                new array, so ``False`` is not supported.

        Returns:
            A NumPy array, or a masked array if a fill value is defined.

        Raises:
            ValueError: If ``copy`` is ``False``.
        """
        if copy is False:
            raise ValueError('A compressed array cannot be converted without '
                             'a copy')
        array: Array = numpy.asanyarray(self._array[...], dtype=self.dtype)
        if self._fill_value is not None:
            array = numpy.ma.masked_equal(array, self._fill_value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def _cast(self, obj: Any) -> Any:
        """Convert an object to a value compatible with a NumPy array."""
        if isinstance(obj, CompressedArray):
            #: pylint: disable=protected-access
            return obj._array[...] if self._fill_value is None else \
                numpy.ma.masked_equal(obj._array[...], self._fill_value)
            #: pylint: enable=protected-access
        if isinstance(obj, numpy.ndarray):
            return obj
        if isinstance(obj, dask.array.core.Array):
            return obj.compute()
        if isinstance(obj, (float, int, bool)):
            return obj
        raise TypeError(f'Cannot cast {type(obj)} to array like object')

    def __array_function__(
        self,
        func: Callable,
        _types: Sequence[type],
        args,
        kwargs,
    ) -> Any:
        """Support NumPy functions on compressed arrays.

        Args:
            func: The NumPy function.
            _types: The types of the arguments.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The result of the NumPy function.
        """
        arrays: list[NDArray] = [
            item.__array__() if isinstance(item, CompressedArray) else item
            for item in args
        ]
        return func(*arrays, **kwargs)

    def __array_ufunc__(
        self,
        ufunc: Callable,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Apply a NumPy universal function to the array.

        Args:
            ufunc: The universal function.
            method: The method to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the universal function.
        """
        inputs = [self._cast(arg) for arg in args]

        if method in ['__call__', 'reduce']:
            result = getattr(ufunc, method)(*inputs, **kwargs)
            if numpy.isscalar(result):
                return result
            return result

        return NotImplemented

    def astype(self, astype_dtype=None, **_kwargs) -> Array:
        """Return a copy of the array with the specified data type.

        Args:
            astype_dtype: The target data type.

        Returns:
            The converted array.
        """
        return self.__array__(astype_dtype)


def dask_array_from_compressed_array(
    array: CompressedArray,
    name: str | None = None,
    inline_array=False,
    **kwargs,
) -> dask.array.core.Array:
    """Create a Dask array backed by a compressed array.

    Args:
        array: The compressed array.

    Returns:
        The Dask array.
    """
    chunks: tuple[int, ...] = array.chunks
    if name is None:
        name = 'from-compressed-array-' + dask.base.tokenize(
            array, chunks, **kwargs)
    return dask.array.core.from_array(
        array,
        chunks,  # type: ignore[arg-type]
        name=name,
        inline_array=inline_array)


@dask.array.dispatch.concatenate_lookup.register(CompressedArray)
def _concatenate_compressed_array(arrays, **kwargs) -> NDArray:
    dtype: DType | None = kwargs.get('dtype', None)
    arr: list[NDArray] = [item.__array__(dtype=dtype) for item in arrays]
    if any(tuple(item.fill_value is not None for item in arrays)):
        return numpy.ma.concatenate(arr, **kwargs)
    return numpy.concatenate(arr, **kwargs)


@dask.array.dispatch.numel_lookup.register(CompressedArray)
def _numel_compressed_array(array, **kwargs) -> numpy.float64 | NDArray:
    #: pylint: disable=protected-access
    arr: NDArray = array.__array__(dtype=kwargs.get('dtype', None))
    if array._fill_value is not None:
        return dask.array.backends._numel_masked(arr, **kwargs)
    return dask.array.backends._numel_ndarray(arr, **kwargs)
    #: pylint: enable=protected-access
