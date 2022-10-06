"""
Overview of a Collection.
=========================

This section outlines the steps required to get started with the main features
of a ``Collection``.
"""
import datetime
import pprint

import dask.distributed
import fsspec

import zcollection
import zcollection.tests.data


# %%
# Initialization of the environment
# ---------------------------------
#
# Before we create our first collection, we will create a dataset to record.
def create_dataset():
    """Create a dataset to record."""
    generator = zcollection.tests.data.create_test_dataset_with_fillvalue()
    return next(generator)


ds = create_dataset()
ds.to_xarray()

# %%
# We will create the file system that we will use. In this example, a file
# system in memory.
fs = fsspec.filesystem('memory')

# %%
# Finally we create a local dask cluster using only threads in order to work
# with the file system stored in memory.
cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

# %%
# Creation of the partitioning
# ----------------------------
#
# Before creating our collection, we define the partitioning of our dataset. In
# this example, we will partition the data by ``month`` using the variable
# ``time``.
partition_handler = zcollection.partitioning.Date(('time', ), resolution='M')

# %%
# Finally, we create our collection:
collection = zcollection.create_collection('time',
                                           ds,
                                           partition_handler,
                                           '/my_collection',
                                           filesystem=fs)

# %%
# .. note::
#
#    The collection created can be accessed using the following command ::
#
#        >>> collection = zcollection.open_collection("/my_collection",
#        >>>                                          filesystem=fs)
#
# When the collection has been created, a configuration file is created. This
# file contains all the metadata to ensure that all future inserted data will
# have the same features as the existing data (data consistency).
pprint.pprint(collection.metadata.get_config())

# %%
# Now that the collection has been created, we can insert new records.
collection.insert(ds)

# %%
# .. note::
#
#     When inserting it's possible to specify the :ref:`merge strategy of a
#     partition <merging_datasets>`. By default, the last inserted data
#     overwrite the existing ones. Others strategy can be defined, for example,
#     to update existing data (overwrite the updated data, while keeping the
#     existing ones). This last strategy allows updating incrementally an
#     existing partition. ::
#
#         >>> import zcollection.merging
#         >>> collection.insert(
#         ...     ds, merge_callable=zcollection.merging.merge_time_series)
#
# Let's look at the different partitions thus created.
pprint.pprint(fs.listdir('/my_collection/year=2000'))

# %%
# This collection is composed of several partitions, but it is always handled
# as a single data set.
#
# Loading data
# ------------
# To load the dataset call the method
# :py:meth:`load<zcollection.collection.Collection.load>` on the instance.  By
# default, the method loads all partitions stored in the collection.
collection.load()

# %%
# You can also filter the partitions to be considered by filtering the
# partitions using keywords used for partitioning in a valid Python expression.
collection.load(filters='year == 2000 and month == 2')

# %%
# You can also used a callback function to filter partitions with a complex
# condition.
collection.load(
    filters=lambda keys: datetime.date(2000, 2, 15) <= datetime.date(
        keys['year'], keys['month'], 1) <= datetime.date(2000, 3, 15))

# %%
# Note that the :py:meth:`load<zcollection.collection.Collection.load>`
# function may return None if no partition has been selected.
assert collection.load(filters='year == 2002 and month == 2') is None

# %%
# Editing variables
# -----------------
#
# .. note::
#
#     The functions for modifying collections are not usable if the collection
#     is :py:meth:`open<zcollection.open_collection>` in read-only mode.
#
# It's possible to delete a variable from a collection.
collection.drop_variable('var2')
collection.load()

# %%
# The variable used for partitioning cannot be deleted.
try:
    collection.drop_variable('time')
except ValueError as exc:
    print(exc)

# %%
# The :py:meth:`add_variable<zcollection.collection.Collection.add_variable>`
# method allows you to add a new variable to the collection.
collection.add_variable(ds.metadata().variables['var2'])

# %%
# The newly created variable is initialized with its default value.
ds = collection.load()
assert ds is not None
ds.variables['var2'].values


# %%
# Finally it's possible to
# :py:meth:`update<zcollection.collection.Collection.update>` the existing
# variables.
#
# In this example, we will alter the variable ``var2`` by setting it to 1
# anywhere the variable ``var1`` is defined.
def ones(ds):
    """Returns a variable with ones everywhere."""
    return dict(var2=ds.variables['var1'].values * 0 + 1)


collection.update(ones)
ds = collection.load()
assert ds is not None
ds.variables['var2'].values

# %%
# Map a function over the collection
# ----------------------------------
# It's possible to map a function over the partitions of the view.
for partition, array in collection.map(
        lambda ds: ds['var1'].values + ds['var2'].values).compute():
    print(f' * partition = {partition}: mean = {array.mean()}')

# %%
# Close the local cluster to avoid printing warning messages in the other
# examples.
client.close()
cluster.close()
