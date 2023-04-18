# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Array indexer
=============

Tools to index Zarr arrays.
"""
from __future__ import annotations

from typing import Any, Sequence, Union
import abc
import dataclasses

import numpy
import zarr

from ..type_hints import NDArray

#: The type of key for indexing.
Key = Union[slice, NDArray]


@dataclasses.dataclass(frozen=True)
class ShiftIndexer:
    """Properties of the slice offset when the user selects a subset of the
    partitions."""
    #: Axis along which the slice is shifted.
    axis: int

    #: Number of items to shift the slice by.
    offset: int


def _expand_indexer(
    key: Any,
    shape: Sequence[int],
) -> tuple[slice, ...]:
    """Given a key for indexing, return an equivalent key which is a tuple with
    length equal to the number of dimensions of the array."""
    ndim = len(shape)
    if not isinstance(key, tuple):
        key = (key, )
    result = []
    found_ellipsis = False
    for item in key:
        if item is Ellipsis:
            if not found_ellipsis:
                result.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                result.append(slice(None))
        else:
            result.append(item)
    if len(result) > ndim:
        raise IndexError(
            f'too many indices for array: array is {ndim}-dimensional,'
            f'but {len(result)} were indexed')
    result.extend((ndim - len(result)) * [slice(None)])
    for ix, dim in enumerate(shape):
        item = result[ix]
        result[ix] = slice(item.start or 0, item.stop or dim, item.step or 1)
    return tuple(result)


def _calculate_tensor_domain(
        key: Sequence[slice],
        chunked_axis: ShiftIndexer | None = None) -> Sequence[slice]:
    """Calculation of the slice of the selected indexes along the chunked axis.

    Args:
        key: The key for indexing.
        offset: The properties of the chunked axis along which the slice is
            shifted.

    Returns:
        The active slice for the given key.
    """
    result = list(slice(item.start, item.stop, None) for item in key)
    if chunked_axis is not None and chunked_axis.offset != 0:
        axis = chunked_axis.axis
        result[axis] = slice(
            result[axis].start - chunked_axis.offset,
            result[axis].stop - chunked_axis.offset,
            None,
        )
    return tuple(result)


def _calculate_array_step(key: NDArray) -> int:
    """Calculate the step of a vectorized indexer."""
    if key.size < 2:
        return 1
    return key[1] - key[0]


def _shift_to_tensor_domain(
    key: Sequence[slice],
    tensor_domain: Sequence[Key],
) -> Sequence[slice]:
    """Shift the slice to the tensor domain.

    Args:
        key: The indexer to shift.
        tensor_domain: The interval defining the validity range of the tensor
            indexed.

    Returns:
        The shifted indexer.
    """

    # If one of the offsets is relative to the size of the vector (negative
    # values), then we need to convert it to an offset relative to its
    # beginning.
    def relative_to_start(item: int | None, indexer: Key) -> int | None:
        """Convert an offset relative to the end of the vector to an offset
        relative to the beginning of the vector."""
        size: int = indexer.stop if isinstance(indexer,
                                               slice) else indexer.size
        return item + size if item is not None and item < 0 else item

    key = tuple(
        slice(
            relative_to_start(new.start, previous),
            relative_to_start(new.stop, previous),
            new.step,
        ) for previous, new in zip(tensor_domain, key))

    # Then we can adjust the start and stop to the valid range.
    def shift_start(previous: Key, new: int | None) -> int:
        """Shift the start of the slice to the valid range."""
        start: int = previous.start if isinstance(previous, slice) else 0
        return max(start, new) if new is not None else start

    def shift_stop(previous: Key, new: int | None) -> int:
        """Shift the stop of the slice to the valid range."""
        stop: int = previous.stop if isinstance(previous,
                                                slice) else len(previous)
        return min(stop, new) if new is not None else stop

    return tuple(
        slice(
            shift_start(previous, new.start),
            shift_stop(previous, new.stop),
            new.step,
        ) for previous, new in zip(tensor_domain, key))


@dataclasses.dataclass(frozen=True)
class Indexer(abc.ABC):
    """Base class for indexing into a variable.

    Args:
        key: Key used for indexing.
        start: The start index.
        stop: The stop index.
    """
    key: Sequence[slice | None]
    tensor_domain: Sequence[Key]

    @abc.abstractmethod
    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Return the chunked array for the given indexer.

        Args:
            array: The chunked array to index.
            axis: The axis along which the concatenation is performed.

        Returns:
            The values of the array read from the given indexer.
        """


@dataclasses.dataclass(frozen=True)
class EmptySelection(Indexer):
    """Indexer used when no chunk is selected."""

    def getitem(self, array: zarr.Array, axis: int = 0) -> NDArray:
        """Return the array sliced by the indexer."""
        # To be consistent with numpy, we return an empty array with the
        # concatenated dimension equal to zero.
        return numpy.empty(expected_shape(array.shape, axis, 0),
                           dtype=array.dtype)


@dataclasses.dataclass(frozen=True)
class SingleSelection(Indexer):
    """Indexer used on a single selected chunk."""

    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Return the array sliced by the indexer."""
        return array[axis][self.key]


@dataclasses.dataclass(frozen=True)
class MultipleSelection(Indexer):
    """Indexer used when several chunks are selctioned."""

    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Return the array sliced by the indexer."""
        data: list[NDArray] = [
            chunk[key] for chunk, key in zip(array, self.key)
            if key is not None
        ]
        return numpy.concatenate(data, axis=axis)


def _calculate_previous_selection_step(tensor_domain: Sequence[Key] | None,
                                       axis: int) -> int:
    """Calculate the step of the previous selection applied on the chunked
    tensor."""
    if tensor_domain is not None:
        item = tensor_domain[axis]
        if isinstance(item, slice):
            return item.step or 1
        return _calculate_array_step(item)
    return 1


def _calculate_vectorized_indexer(
    axis: int,
    chunk_size: NDArray,
    chunks: Sequence[zarr.Array],
    ix: int,
    key: slice,
    keys: Sequence[slice],
    ndim: int,
    previous_selection_step: int,
    tensor_domain: Sequence[Key] | None,
) -> Indexer:
    """Calculate a vector of indices for the selected chunks."""
    indices: NDArray = numpy.arange(key.start, key.stop)
    if tensor_domain is not None and previous_selection_step != 1:
        item = tensor_domain[axis]
        # If the previous selection step is not 1, item is a vector of
        # indices
        assert isinstance(item, numpy.ndarray)
        indices = item[indices]

    indices = indices[::key.step]

    whole_selected_indices: NDArray = indices - calculate_offset(
        chunk_size, ix)
    chunked_indices: list[Any] = [None] * len(chunks)
    while indices.size > 0:
        mask: NDArray = indices < chunk_size[ix]
        chunked_indices[ix] = indices[mask] - calculate_offset(chunk_size, ix)
        indices = indices[~mask]
        ix += 1

    return MultipleSelection(
        chunked_indices,
        tuple(whole_selected_indices if ix == axis else keys[ix]
              for ix in range(ndim)))


def expected_shape(shape: Sequence[int],
                   axis: int,
                   value: int = -1) -> Sequence[int]:
    """Return the expected shape of a variable after concatenation."""
    shape = tuple(shape)
    return shape[:axis] + (value, ) + shape[axis + 1:]


def slice_length(key: Key, length: int) -> int:
    """Calculate the length of the slice.

    Args:
        key: The slice to calculate the length of.
        length: The length of the array.

    Returns:
        The length of the slice.
    """
    # To properly calculate the size of the slice, even if the indexes are
    # relative to the end of the array, we use the range function.
    if isinstance(key, slice):
        return len(range(key.start or 0, key.stop or length, key.step or 1))
    return ((key[-1] - key[0]) // _calculate_array_step(key)) + 1


def calculate_shape(key: Sequence[Key] | None,
                    shape: Sequence[int]) -> Sequence[int]:
    """Calculate the shape of the array for the given indexer."""
    if key is None:
        return shape
    return tuple(
        slice_length(key_item, shape_item)
        for key_item, shape_item in zip(key, shape))


def calculate_offset(chunk_size: NDArray, ix: Any) -> int:
    """Calculate the offset of the given chunk."""
    return chunk_size[ix - 1] if ix > 0 else 0  # type: ignore[return-value]


def get_indexer(
    chunks: Sequence[zarr.Array],
    shape: Sequence[int],
    key: Any,
    tensor_domain: Sequence[Key] | None,
    axis: int = 0,
) -> Indexer:
    """Get the indexer for the given key.

    Args:
        chunks: The chunked array to index.
        ndim: The number of dimensions of the chunked array.
        key: The key used for indexing.
        tensor_domain: The interval defining the validity range of the tensor
            indexing.
        axis: The chunked axis.

    Returns:
        The indexer for the given key.
    """
    key = _expand_indexer(key, shape)

    # If the array is a view on an existing chunked array, we need to restrict
    # the indexer on the view bounds.
    if tensor_domain is not None:
        key = _shift_to_tensor_domain(key, tensor_domain)

    if len(chunks) == 1:
        return SingleSelection(key, _calculate_tensor_domain(key))

    # Reference to the chunked slice.
    chunked_key = key[axis]

    # Calculate the cumulative shape of the chunks.
    chunk_size = numpy.cumsum(
        numpy.array(tuple(item.shape[axis] for item in chunks)))

    # Start index on the chunked axis.
    start = chunked_key.start or 0
    if start < 0:
        start = chunk_size[-1] + start

    # Stop index on the chunked axis.
    stop = min(chunked_key.stop or chunk_size[-1], chunk_size[-1])
    if stop < 0:
        stop = chunk_size[-1] + stop

    # Slice step on the chunked axis.
    step = chunked_key.step or 1

    # The result of an invalid selection is an empty array.
    if start >= stop:
        return EmptySelection(key, _calculate_tensor_domain(key))

    # First index on the chunked axis.
    ix0 = int(numpy.searchsorted(chunk_size, start, side='right'))

    # Last index on the chunked axis.
    ix1: int = min(
        len(chunk_size) - 1,
        numpy.searchsorted(chunk_size, stop, side='left'),
    )

    # Calculate the previous selection step applied on the chunked tensor.
    previous_selection_step: int = _calculate_previous_selection_step(
        tensor_domain, axis)

    # If the indexing step, current or previous, is different from 1, we have
    # to use a vectorized indexer to correctly handle a non-uniform chunked
    # array.
    if step != 1 or previous_selection_step != 1:
        return _calculate_vectorized_indexer(
            axis,
            chunk_size,
            chunks,
            ix0,
            slice(start, stop, step),
            key,
            len(shape),
            previous_selection_step,
            tensor_domain,
        )

    # Move the start index forward to the selected chunk.
    start -= calculate_offset(chunk_size, ix0)

    # Move the stop index forward to the selected chunk.
    stop -= calculate_offset(chunk_size, ix1)

    # Build the key for selecting each chunk.
    chunk_slice: list[slice | None] = [None] * len(chunks)

    if ix0 == ix1:
        # Only one chunk is selected.
        chunk_slice[ix0] = slice(start, stop, None)
    else:
        chunk_slice[ix0] = slice(start, None, None)
        chunk_slice[ix1] = slice(None, stop, None)
        chunk_slice[ix0 + 1:ix1] = [slice(None)] * (ix1 - ix0 - 1)

    return MultipleSelection(
        chunk_slice,
        _calculate_tensor_domain(
            key,
            ShiftIndexer(axis, calculate_offset(chunk_size, ix0)),
        ),
    )
