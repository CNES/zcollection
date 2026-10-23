# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Reports of a migration
======================

What a run of a migration function did to the partitions it selected, and
how far the copy of a collection has got.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What one run did to the partitions it selected.

    The partitions are named by their path relative to their collection, and
    each field is sorted. The documentation of each function states what it
    puts in each field.

    Each field records what a task decided, not what the two collections
    hold once the run returns. Dask runs a task again when the worker
    holding its result dies, so a partition one run wrote can come back
    under ``skipped``. :func:`status <zcollection.migration.status>` and
    :func:`check <zcollection.migration.check>` are what read the storage.
    """
    #: The partitions the run reports as done.
    done: tuple[str, ...]

    #: The partitions it left as they were.
    skipped: tuple[str, ...]

    #: The partitions it reports as failed, each with the reason as text.
    failed: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class Status:
    """How far the copy of a collection has got."""
    #: The partitions both collections hold, complete in the target.
    migrated: tuple[str, ...]

    #: The partitions of the source the target does not hold complete.
    missing: tuple[str, ...]

    #: The partitions the target holds complete and the source no longer
    #: holds.
    pruned: tuple[str, ...]

    #: The partitions the source no longer holds and the target does not
    #: hold complete. Nothing can rebuild them.
    lost: tuple[str, ...]

    def __str__(self) -> str:
        """Return the counts grouped by the first partitioning key.

        Returns:
            One line per value of that key, in the order the values sort.
            The pruned and the lost counts are left out of a group holding
            none.
        """
        groups: dict[str, list[int]] = {}
        for index, partitions in enumerate(
            (self.migrated, self.missing, self.pruned, self.lost)):
            for item in partitions:
                groups.setdefault(item.split('/')[0], [0, 0, 0, 0])[index] += 1

        lines = []
        for key in sorted(groups):
            migrated, missing, pruned, lost = groups[key]
            line = f'{key} : {migrated} migrated, {missing} missing'
            if pruned:
                line += f', {pruned} pruned'
            if lost:
                line += f', {lost} lost'
            lines.append(line)
        return '\n'.join(lines)
