# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test partitioning by date.
==========================
"""
from typing import Iterator
import pickle

import dask.array
import numpy
import pytest
import xarray

from .. import Date, get_codecs
from ... import dataset


def test_split_dataset():
    """Test the split_dataset method"""
    start_date = numpy.datetime64("2000-01-06", "us")
    delta = numpy.timedelta64(1, "h")

    for end_date, indices, resolution in [
        (
            numpy.datetime64("2001-12-31", "Y"),
            slice(0, 1),
            "Y",
        ),
        (
            numpy.datetime64("2000-12-31", "M"),
            slice(0, 2),
            "M",
        ),
        (
            numpy.datetime64("2000-12-31", "D"),
            slice(0, 3),
            "D",
        ),
        (
            numpy.datetime64("2000-01-31", "h"),
            slice(0, None),
            "h",
        ),
    ]:

        # Time delta between two partitions
        timedelta = numpy.timedelta64(1, resolution)

        # Temporal axis to split
        dates = numpy.arange(start_date, end_date, delta)

        # Measured data
        observation = numpy.random.rand(dates.size)  # type: ignore

        # Create the dataset to split
        ds = xarray.Dataset(
            dict(dates=xarray.DataArray(dates, dims=("num_lines", )),
                 observation=xarray.DataArray(observation,
                                              dims=("num_lines", ))))

        partitioning = Date(("dates", ), resolution)

        # Date of the current partition
        date = numpy.datetime64(start_date, resolution)

        # Build the test dataset
        ds = dataset.Dataset.from_xarray(ds)

        iterator = partitioning.split_dataset(ds, "num_lines")
        assert isinstance(iterator, Iterator)

        for partition, indexer in iterator:
            subset = ds.isel(indexer)

            # Cast the date to the a datetime object to extract the date
            item = date.astype("datetime64[us]").item()
            expected = (
                f"year={item.year}",
                f"month={item.month:02d}",
                f"day={item.day:02d}",
                f"hour={item.hour:02d}",
            )
            assert partition == expected[indices]

            folder = "/".join(partition)
            fields = partitioning.parse(folder)
            parsed_date, = partitioning.encode(fields)
            assert parsed_date == numpy.datetime64(date).astype(
                f"datetime64[{resolution}]")

            expected_selection = dates[
                (dates >= parsed_date)  # type: ignore
                & (dates < parsed_date + timedelta)]  # type: ignore
            assert numpy.all(
                subset.variables["dates"].array == expected_selection)

            expected = (
                ("year", item.year),
                ("month", item.month),
                ("day", item.day),
                ("hour", item.hour),
            )
            assert fields == expected[indices]
            assert partitioning.join(fields, "/") == folder
            assert partitioning.join(partitioning.decode((parsed_date, )),
                                     "/") == folder

            date += timedelta


def test_construction():
    """Test the construction of the Date class."""
    partitioning = Date(("dates", ), "D")
    assert partitioning.resolution == "D"
    assert partitioning.variables == ("dates", )
    assert partitioning.dtype() == (("year", "uint16"), ("month", "uint8"),
                                    ("day", "uint8"))
    assert partitioning.get_config() == {
        "id": "Date",
        "resolution": "D",
        "variables": ("dates", ),
    }

    with pytest.raises(ValueError):
        Date(("dates1", "dates2"), "D")

    with pytest.raises(ValueError):
        Date(("dates", ), "W")


def test_config():
    """Test the configuration of the Date class."""
    partitioning = Date(("dates", ), "D")
    assert partitioning.dtype() == (("year", "uint16"), ("month", "uint8"),
                                    ("day", "uint8"))
    config = partitioning.get_config()
    partitioning = get_codecs(config)
    assert isinstance(partitioning, Date)


def test_pickle():
    """Test the pickling of the Date class."""
    partitioning = Date(("dates", ), "D")
    other = pickle.loads(pickle.dumps(partitioning))
    assert isinstance(other, Date)
    assert other.resolution == "D"
    assert other.variables == ("dates", )


def test_no_monotonic():
    """Test that the Date partitioning raises an error if the temporal axis is
    not monotonic."""
    dates = numpy.arange(numpy.datetime64("2000-01-01", "h"),
                         numpy.datetime64("2000-01-02", "h"),
                         numpy.timedelta64(1, "m"))
    numpy.random.shuffle(dates)
    partitioning = Date(("dates", ), "h")
    with pytest.raises(ValueError):
        list(partitioning._split({"dates": dask.array.from_array(dates)}))


def test_values_must_be_datetime64():
    """Test that the values must be datetime64."""
    dates = numpy.arange(numpy.datetime64("2000-01-01", "h"),
                         numpy.datetime64("2000-01-02", "h"),
                         numpy.timedelta64(1, "m"))
    partitioning = Date(("dates", ), "h")
    dates = dates.astype("int64")
    with pytest.raises(TypeError):
        list(partitioning._split({"dates": dask.array.from_array(dates)}))
