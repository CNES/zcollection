"""
Overview of a View.
===================

This section outlines the steps required to get started with the main features
of a ``View``.
"""
import pprint

import dask.distributed
import fsspec
import numpy

import zcollection
import zcollection.tests.data


# %%
# Initialization of the environment
# ---------------------------------
#
# As in the example of handling
# :ref:`collections <sphx_glr_auto_examples_ex_collection.py>`, we will create
# the test environment and a collection.
def create_dataset():
    """Create a dataset to record."""
    generator = zcollection.tests.data.create_test_dataset_with_fillvalue()
    return next(generator)


cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

ds = create_dataset()
fs = fsspec.filesystem('memory')
collection = zcollection.create_collection('time',
                                           ds,
                                           zcollection.partitioning.Date(
                                               ('time', ), resolution='M'),
                                           '/view_reference',
                                           filesystem=fs)
collection.insert(ds, merge_callable=zcollection.merging.merge_time_series)

# %%
# Creation of views
# -----------------
#
# A :py:class:`view<zcollection.view.View>` allows you to extend a collection
# (:py:class:`a view reference<zcollection.view.ViewReference>`) that you are
# not allowed to modify.
view = zcollection.create_view('/my_view',
                               zcollection.view.ViewReference(
                                   '/view_reference', fs),
                               filesystem=fs)

# %%
# .. note::
#
#     The created view can be accessed using the following command ::
#
#         >>> view = zcollection.open_view("/my_view", filesystem=fs)
#
# Editing variables
# -----------------
# When the view is created, it has no data of its own, it uses all the
# partitions defined in the reference view. You can select the partitions used
# from the reference collection by specifying the keyword argument ``filters``
# during the creation of the view.
pprint.pprint(fs.listdir('/my_view'))

# %%
# It's not yet possible to read data from the view, as it does not yet have any
# data. To minimize the risk of mismatches with the reference view, the data
# present in the view drives the range of data that can be read.
try:
    view.load()
except ValueError as err:
    print(err)

# %%
# Such a state of the view is not very interesting. But it is possible to
# :py:meth:`add<zcollection.view.View.add_variable>` and modify variables in
# order to enhance the view.
var3 = ds.metadata().variables['var2']
var3.name = 'var3'
view.add_variable(var3)

# %%
# This step creates all necessary partitions for the new variable.
pprint.pprint(fs.listdir('/my_view/year=2000'))

# %%
# The new variable is not initialized.
ds = view.load()
assert ds is not None
ds.variables['var3'].values

# %%
# The same principle used by the collection allows to
# :py:meth:`update<zcollection.view.View.update>` the variables.
view.update(
    lambda ds: dict(var3=ds['var1'].values * 0 + 1))  # type: ignore[arg-type]

# %%
# Like the :py:meth:`update<zcollection.collection.Collection.update>` method
# of the collection, the update method of view allows to selecting the
# neighboring partitions with the keyword argument ``depth``.

# %%
ds = view.load()
assert ds is not None
var3 = ds['var3'].values
var3

# %%
# **Warning**: The variables of the reference collection cannot be edited.
try:
    view.update(
        lambda ds: dict(var2=ds['var2'].values * 0))  # type: ignore[arg-type]
except ValueError as exc:
    print(str(exc))


# %%
# Sync the view with the reference
# --------------------------------
# The view may not be read anymore if the number of elements in the reference
# collection and in the view is not identical. To avoid this problem, the view
# is automatically synchronized when it is opened. But only if the reference
# collection has been completed (adding new data after the existing data), the
# data already present in the view are kept. The existing tables in the view are
# resized and filled with the defined fill values. If you want to know which
# partitions are synchronized, you have to use the following data flow: open the
# view and ask not to synchronize it (``resync=False``), then call the ``sync``
# method of view class to obtain a filter allowing selecting all the partitions
# that have been modified.
#
# Let's illustrate this data flow with an example.
#
# First, we create an utility function to resize a dataset.
def resize(ds: zcollection.Dataset, dim: str,
           size: int) -> zcollection.Dataset:
    """Resize a dataset."""

    def new_shape(
        var: zcollection.Variable,
        selected_dim: str,
        new_size: int,
    ) -> tuple[int, ...]:
        """Compute the new shape of a variable."""
        return tuple(new_size if dim == selected_dim else size
                     for dim, size in zip(var.dimensions, var.shape))

    return zcollection.Dataset([
        zcollection.Variable(
            name,
            numpy.resize(var.array.compute(), new_shape(var, dim, size)),
            var.dimensions,
            var.attrs,
            var.compressor,
            var.fill_value,
            var.filters,
        ) for name, var in ds.variables.items()
    ])


# %%
# We then modify the last partition of the reference collection. We start by
# opening the reference collection and loading the last partition.
collection = zcollection.open_collection('/view_reference',
                                         filesystem=fs,
                                         mode='w')
ds = collection.load(
    filters=lambda keys: keys['month'] == 6 and keys['year'] == 2000)
assert ds is not None

# %%
# We create a new time variable, resize the dataset and insert the new time
# values.
time: numpy.ndarray = numpy.arange(
    numpy.datetime64('2000-06-01T00:00:00'),
    numpy.datetime64('2000-06-30T23:59:59'),
    numpy.timedelta64(1, 'h'),
)
ds = resize(ds, 'num_lines', len(time))
ds['time'].data = time

# %%
# Finally, we update the partition in the reference collection.
collection.insert(ds)

# %%
# Now we cannot load the view, because the shape of the last partition is no
# longer consistent between the reference collection and the view.
try:
    view.load()
except ValueError as err:
    print(err)

# %%
# We call the ``sync`` method to resynchronize the view.
filters = view.sync()

# %%
# The method returns a callable that can be used to filter the partitions that
# have been synchronized. You can use this information to perform a
# :py:meth:`update<zcollection.view.View.update>` of the view on the
# synchronized partitions: ::
#
#     view.update(
#         lambda ds: dict(var3=ds['var1'].values * 0 + 1),
#         filters=filters)
#
print(tuple(view.partitions(filters=filters)))

# %%
# The view is now synchronized and can be loaded.
ds = view.load()
assert ds is not None
ds.variables['var3'].values

# %%
# Map a function over the view
# ----------------------------
# It's possible to map a function over the partitions of the view.
for partition, array in view.map(lambda ds: (  # type: ignore[arg-type]
        ds['var1'].values + ds['var2'].values)).compute():
    print(f' * partition = {partition}: mean = {array.mean()}')

# %%
# .. seealso::
#
#     See the :py:meth:`map_overlap<zcollection.view.View.map_overlap>` method
#     apply a function over the partitions of the view of selecting the
#     neighboring partitions.
#
# Drop a variable
# ----------------
# A method allows you to
# :py:meth:`drop_variable<zcollection.view.View.drop_variable>` variables from
# the view.
view.drop_variable('var3')
try:
    view.load()
except ValueError as err:
    # The view no longer has a variable.
    print(err)

# %%
# **Warning**: The variables of the reference collection cannot be dropped.
try:
    view.drop_variable('var2')
except ValueError as exc:
    print(str(exc))

# %%
# Close the local cluster to avoid printing warning messages in the other
# examples.
client.close()
cluster.close()
