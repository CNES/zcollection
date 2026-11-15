# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Reports of a migration
======================

Outcome records describe individual migration runs. Status records
compare the partitions available in the source and target.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Results of a migration operation on selected partitions.

    Partition paths are relative to their collection, and each field is
    sorted. The meaning of each field depends on the operation.

    Fields report task outcomes rather than the final state of storage.
    If Dask retries a task after a worker failure, a partition written
    during an earlier attempt may be reported as ``skipped``. Use
    :func:`status <zcollection.migration.status>` or
    :func:`check <zcollection.migration.check>` to inspect storage.
    """
    #: The partitions the run reports as done.
    done: tuple[str, ...]

    #: The partitions it left as they were.
    skipped: tuple[str, ...]

    #: The partitions it reports as failed, each with the reason as text.
    failed: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class Status:
    """Migration status of a collection or view."""
    #: Partitions present in the source and complete in the target.
    migrated: tuple[str, ...]

    #: Source partitions missing or incomplete in the target.
    missing: tuple[str, ...]

    #: Partitions absent from the source but complete in the target,
    #: usually because ``prune`` removed them.
    pruned: tuple[str, ...]

    #: Partitions absent from the source and incomplete in the target.
    #: Neither collection has a complete copy. ``prune`` never removes
    #: these partitions; this can happen if a source partition is
    #: deleted before its copy is complete.
    lost: tuple[str, ...]

    def __str__(self) -> str:
        """Return migration counts grouped by the first partitioning key.

        Returns:
            One line per key value, sorted by that value. Counts of zero
            for ``pruned`` and ``lost`` are omitted.
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
