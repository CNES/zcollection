"""
Overview of a View.
===================

This section outlines the steps required to get started with the main features
of a ``Collection``.
"""
import pprint

import dask.distributed
import fsspec

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
    """Create a dataset to record
    """
    generator = zcollection.tests.data.create_test_dataset_with_fillvalue()
    return next(generator)


ds = create_dataset()
fs = fsspec.filesystem('memory')
cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)
collection = zcollection.create_collection("time",
                                           ds,
                                           zcollection.partitioning.Date(
                                               ("time", ), resolution="M"),
                                           "/my_collection",
                                           filesystem=fs)
collection.insert(ds, merge_callable=zcollection.merging.merge_time_series)

# %%
# Creation of views
# -----------------
#
# A :py:class:`view<zcollection.view.View>` allows you to extend a collection
# (:py:class:`a view reference<zcollection.view.ViewReference>`) that you are
# not allowed to modify.
view = zcollection.create_view("/my_view",
                               zcollection.view.ViewReference(
                                   "/my_collection", fs),
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
# When the view is created, it has no data of its own, it uses all the data
# defined in the reference view.
pprint.pprint(fs.listdir("/my_view"))

# %%
# But you can load the data of the view using the following command:
view.load()

# %%
# Such a state of the view is not very interesting. But it is possible to
# :py:meth:`add<zcollection.view.View.add_variable>` and modify variables in
# order to enhance the view.
var3 = ds.metadata().variables["var2"]
var3.name = "var3"
view.add_variable(var3)

# %%
# This step creates all necessary partitions for the new variable.
pprint.pprint(fs.listdir("/my_view/year=2000"))

# %%
# The new variable is not initialized.
ds = view.load()
assert ds is not None
ds.variables["var3"].values

# %%
# The same principle used by the collection allows to
# :py:meth:`update<zcollection.view.View.update>` the variables.
view.update(lambda ds: ds["var1"].values * 0 + 1, "var3")

# %%
ds = view.load()
assert ds is not None
var3 = ds["var3"].values
var3

# %%
var2 = ds["var2"].values
var2

# %%
var2 - var3

# *Warning*: The variables of the reference collection cannot be edited.
try:
    view.update(lambda ds: ds["var2"].values * 0, "var2")
except ValueError as exc:
    print(str(exc))

# %%
# Finally, a method allows you to
# :py:meth:`drop_variable<zcollection.view.View.drop_variable>` variables from
# the view.
view.load()

# %%
view.drop_variable("var3")
view.load()

# %%
# *Warning*: The variables of the reference collection cannot be dropped.
try:
    view.drop_variable("var2")
except ValueError as exc:
    print(str(exc))

# %%
# Close the local cluster to avoid printing warning messages in the other
# examples.
client.close()
cluster.close()
