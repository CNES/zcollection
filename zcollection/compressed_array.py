# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Compressed array class
======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.array.backends
import dask.array.chunk_types
import dask.array.core
import dask.array.dispatch
import dask.base
import numpy
import numpy.lib.mixins
import zarr
import zarr.abc.codec

from .type_hints import DType, NDArray, NDMaskedArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Type of arrays returned when a compressed array is decompressed.
Array = NDArray | NDMaskedArray


class CompressedArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    """Hold a compressed array and provide a numpy.ndarray interface to it.

    Each operation on the array uncompresses the data, performs the operation,
    and returns a numpy array. A rechunk holds the new chunks behind the
    codecs of the array it is built from.

    Data is compressed using zarr.

    Args:
        array: A numpy.ndarray or a zarr.Array.
        **kwargs: Keyword arguments to pass to zarr.create_array.
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
    ) -> Array:
        """Retrieve data for an item or region of the array.

        Args:
            key: The key to retrieve.

        Returns:
            The data for the key.
        """
        values: Array = self._array[key]
        if self._fill_value is not None:
            values = numpy.ma.masked_equal(values, self._fill_value)
        return values

    def __setitem__(
        self,
        key: int | tuple[int | slice, ...],
        value: Array,
    ) -> None:
        """Set data for an item or region of the array.

        Args:
            key: The key to set.
            value: The value to set.
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
    def compressors(self) -> tuple[zarr.abc.codec.Codec, ...]:
        """The codecs the compressed chunks are held behind."""
        return self._array.compressors

    @property
    def fill_value(self) -> Any:
        """The value used for uninitialized portions of the array."""
        return self._fill_value

    @property
    def filters(self) -> tuple[zarr.abc.codec.Codec, ...]:
        """The codecs applied to the values before compression."""
        return self._array.filters

    def reshape(self, *shape: int) -> Array:
        """Return a new array with the same data with a new shape.

        Args:
            *shape: The new shape.

        Returns:
            The reshaped array.
        """
        return self.__array__().reshape(*shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        """The length of each dimension of a chunk of the array."""
        return self._array.chunks

    def rechunk(self, chunks: tuple[int, ...]) -> CompressedArray:
        """Rechunk the array.

        Args:
            chunks: The new chunks.

        Returns:
            The rechunked array.
        """
        return CompressedArray(self._array[...],
                               chunks=chunks,
                               compressors=self._array.compressors,
                               filters=self._array.filters,
                               serializer=self._array.serializer,
                               fill_value=self._fill_value)

    def __array__(self, dtype=None) -> Array:
        """Return a numpy.ndarray of the array.

        Args:
            dtype: The data type of the array.

        Returns:
            The numpy.ndarray of the array or numpy.ma.masked_array if
            a fill_value is defined.
        """
        array: Array = self._array[...]
        if self._fill_value is not None:
            array = numpy.ma.masked_equal(array, self._fill_value)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def _cast(self, obj: Any) -> Any:
        """Cast an object to a value compatible with numpy.ndarray."""
        if isinstance(obj, CompressedArray):
            #: pylint: disable=protected-access
            # obj is a CompressedArray, we can access its _array attribute.
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
        """Support numpy functions.

        Args:
            func: The numpy function.
            _types: The types of the arguments.
            args: The arguments.
            kwargs: The keyword arguments.

        Returns:
            The result of the numpy function.
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
        """Call a numpy ufunc on the array.

        Args:
            ufunc: The ufunc to call.
            method: The method to call.
            *args: The arguments to pass to the ufunc.
            **kwargs: The keyword arguments to pass to the ufunc.

        Returns:
            The result of the ufunc.
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
            astype_dtype: The data type to cast to.

        Returns:
            The casted array.
        """
        return self.__array__(astype_dtype)


def dask_array_from_compressed_array(
    array: CompressedArray,
    name: str | None = None,
    inline_array=False,
    **kwargs,
) -> dask.array.core.Array:
    """Create a dask array from a compressed array.

    Args:
        array: The compressed array.

    Returns:
        The dask array.
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
    # array is a CompressedArray, we can access its _array attribute.
    # We reuse the implementation of dask array numel. Useless to
    # reimplement it.
    arr: NDArray = array.__array__(dtype=kwargs.get('dtype', None))
    if array._fill_value is not None:
        return dask.array.backends._numel_masked(arr, **kwargs)
    return dask.array.backends._numel_ndarray(arr, **kwargs)
    #: pylint: enable=protected-access
