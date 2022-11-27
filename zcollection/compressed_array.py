"""
Compressed array class
======================
"""
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import dask.array.backends
import dask.array.chunk_types
import dask.array.core
import dask.array.dispatch
import dask.base
import numcodecs.abc
import numpy
import numpy.lib.mixins
import zarr

from .type_hints import DType, NDArray, NDMaskedArray

#: Type of arrays returned when a compressed array is decompressed.
Array = Union[NDArray, NDMaskedArray]


class CompressedArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    """Hold a compressed array and provide a numpy.ndarray interface to it.

    Each operation on the array uncompresses the data, performs the operation,
    an returns a numpy array. For the rechunk operation, the array is
    rechunked and the compressor is applied to the new chunks.

    Data is compressed using zarr and numcodecs.

    Args:
        array: A numpy.ndarray or a zarr.Array.
        *args: Arguments to pass to zarr.array.
        **kwargs: Keyword arguments to pass to zarr.array.
    """

    def __init__(
        self,
        array: Union['CompressedArray', NDArray, zarr.Array],
        *args,
        **kwargs,
    ):
        self._fill_value = kwargs.get('fill_value', None)
        if isinstance(array, numpy.ndarray):
            self._array = zarr.array(array, *args, **kwargs)
        elif isinstance(array, zarr.Array):
            if args or kwargs:
                raise ValueError('args or kwargs are not allowed when '
                                 'array is a zarr.Array')
            self._array = array
        elif isinstance(array, CompressedArray):
            self._array = array._array
            self._fill_value = array._fill_value
        else:
            raise TypeError('array must be a numpy.ndarray or a zarr.Array')

    __array_priority__ = 0

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} ({self.shape}) {self.dtype}>'

    def _repr_html_(self) -> str:
        html_code = self._array.info._repr_html_()
        return html_code.replace('zarr.core.Array', str(type(self)))

    def __getitem__(
        self,
        key: Union[int, Tuple[Union[int, slice], ...]],
    ) -> Array:
        """Retrieve data for an item or region of the array.

        Args:
            key: The key to retrieve.

        Returns:
            The data for the key.
        """
        values = self._array[key]
        if self._fill_value is not None:
            values = numpy.ma.masked_equal(values, self._fill_value)
        return values

    def __setitem__(
        self,
        key: Union[int, Tuple[Union[int, slice], ...]],
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
    def shape(self) -> Tuple[int, ...]:
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
    def compressor(self) -> Optional[numcodecs.abc.Codec]:
        """The compressor used to compress the array."""
        return self._array.compressor

    @property
    def fill_value(self) -> Any:
        """The value used for uninitialized portions of the array."""
        return self._fill_value

    @property
    def filters(self) -> Optional[list[numcodecs.abc.Codec]]:
        """One or more codecs used to transform data prior to compression."""
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
    def chunks(self) -> Sequence[Sequence[int]]:
        """A Tuple of integers describing the length of each dimension of a
        chunk of the array."""
        return self._array.chunks

    def rechunk(self, chunks: Sequence[Sequence[int]]) -> 'CompressedArray':
        """Rechunk the array.

        Args:
            chunks: The new chunks.

        Returns:
            The rechunked array.
        """
        return CompressedArray(
            zarr.array(self._array,
                       chunks=chunks,
                       compressor=self.compressor,
                       fill_value=self.fill_value,
                       filters=self.filters))

    def __array__(self, dtype=None) -> Array:
        """Return a numpy.ndarray of the array.

        Args:
            dtype: The data type of the array.

        Returns:
            The numpy.ndarray of the array or numpy.ma.masked_array if
            a fill_value is defined.
        """
        array = self._array[...]
        if self._fill_value is not None:
            array = numpy.ma.masked_equal(array, self._fill_value)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def _cast(self, obj: Any) -> Any:
        """Cast an object to a value compatible with numpy.ndarray."""
        if isinstance(obj, CompressedArray):
            return obj._array[...] if self._fill_value is None else \
                numpy.ma.masked_equal(obj._array[...], self._fill_value)
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
        types: Sequence[Type],
        args,
        kwargs,
    ) -> Any:
        """Support numpy functions.

        Args:
            func: The numpy function.
            types: The types of the arguments.
            args: The arguments.
            kwargs: The keyword arguments.

        Returns:
            The result of the numpy function.
        """
        args = [
            item.__array__() if isinstance(item, CompressedArray) else item
            for item in args
        ]
        return func(*args, **kwargs)

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
    name: Optional[str] = None,
    inline_array=False,
    **kwargs,
) -> dask.array.core.Array:
    """Create a dask array from a compressed array.

    Args:
        array: The compressed array.

    Returns:
        The dask array.
    """
    chunks = array.chunks
    if name is None:
        name = 'from-compressed-array-' + dask.base.tokenize(
            array, chunks, **kwargs)
    return dask.array.core.from_array(
        array,
        chunks,  # type: ignore
        name=name,
        inline_array=inline_array)


@dask.array.dispatch.concatenate_lookup.register(CompressedArray)
def _concatenate_compressed_array(arrays, **kwargs):
    dtype = kwargs.get('dtype', None)
    arr = [item.__array__(dtype=dtype) for item in arrays]
    if any(tuple(item.fill_value is not None for item in arrays)):
        return numpy.ma.concatenate(arr, **kwargs)
    return numpy.concatenate(arr, **kwargs)


@dask.array.dispatch.numel_lookup.register(CompressedArray)
def _numel_compressed_array(array, **kwargs):
    arr = array.__array__(dtype=kwargs.get('dtype', None))
    if array._fill_value is not None:
        return dask.array.backends._numel_masked(arr, **kwargs)
    return dask.array.backends._numel_ndarray(arr, **kwargs)
