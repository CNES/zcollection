"""Implementation of the ``zcollection`` CLI subcommands."""

from typing import Any, TextIO
import argparse
from collections.abc import Sequence
import json
import sys

import zcollection as zc
from zcollection.errors import CollectionNotFoundError, StoreError


def _open_ro(path: str) -> zc.Collection:
    store = zc.open_store(path, read_only=True)
    return zc.open_collection(store, mode="r")


def _open_rw(path: str) -> zc.Collection:
    store = zc.open_store(path)
    return zc.open_collection(store, mode="rw")


def _cmd_ls(args: argparse.Namespace, out: TextIO) -> int:
    col = _open_ro(args.path)
    parts = list(col.partitions(filters=args.filter))
    if args.json:
        json.dump(parts, out)
        out.write("\n")
    else:
        for p in parts:
            out.write(p + "\n")
    return 0


def _cmd_inspect(args: argparse.Namespace, out: TextIO) -> int:
    col = _open_ro(args.path)
    schema = col.schema
    info: dict[str, Any] = {
        "uri": col.store.root_uri,
        "axis": col.axis,
        "partitioning": col.partitioning.to_json(),
        "n_partitions": sum(1 for _ in col.partitions()),
        "dimensions": {
            name: {"size": dim.size, "chunks": dim.chunks}
            for name, dim in schema.dimensions.items()
        },
        "variables": {
            name: {
                "dtype": str(var.dtype),
                "dimensions": list(var.dimensions),
                "immutable": var.immutable,
            }
            for name, var in schema.variables.items()
        },
        "attrs": dict(schema.attrs),
    }
    if args.json:
        json.dump(info, out, indent=2, default=str)
        out.write("\n")
    else:
        _render_inspect(info, out)
    return 0


def _render_inspect(info: dict[str, Any], out: TextIO) -> None:
    out.write(f"uri:          {info['uri']}\n")
    out.write(f"axis:         {info['axis']}\n")
    out.write(f"partitioning: {info['partitioning']}\n")
    out.write(f"partitions:   {info['n_partitions']}\n")
    out.write("dimensions:\n")
    for name, dim in info["dimensions"].items():
        out.write(f"  {name}: size={dim['size']} chunks={dim['chunks']}\n")
    out.write("variables:\n")
    for name, var in info["variables"].items():
        flag = " (immutable)" if var["immutable"] else ""
        dims = ",".join(var["dimensions"])
        out.write(f"  {name}: {var['dtype']} ({dims}){flag}\n")
    if info["attrs"]:
        out.write("attrs:\n")
        for k, v in info["attrs"].items():
            out.write(f"  {k}: {v}\n")


def _cmd_drop(
    args: argparse.Namespace,
    out: TextIO,
    stdin: TextIO,
) -> int:
    col = _open_rw(args.path)
    targets = list(col.partitions(filters=args.filter))
    if not targets:
        out.write("no partitions match\n")
        return 0
    preview_limit = 10
    if not args.yes:
        out.write(f"about to drop {len(targets)} partition(s):\n")
        for p in targets[:preview_limit]:
            out.write(f"  {p}\n")
        if len(targets) > preview_limit:
            out.write(f"  ... ({len(targets) - preview_limit} more)\n")
        out.write("proceed? [y/N] ")
        out.flush()
        reply = stdin.readline().strip().lower()
        if reply not in {"y", "yes"}:
            out.write("aborted\n")
            return 1
    dropped = col.drop_partitions(filters=args.filter)
    out.write(f"dropped {len(dropped)} partition(s)\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zcollection",
        description="Inspect and manage zcollection v3 collections.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("ls", help="List partition paths.")
    ls.add_argument("path", help="Collection URL or filesystem path.")
    ls.add_argument(
        "--filter", default=None, help="Partition filter expression."
    )
    ls.add_argument(
        "--json", action="store_true", help="Output as a JSON list."
    )

    insp = sub.add_parser("inspect", help="Show schema and partition summary.")
    insp.add_argument("path", help="Collection URL or filesystem path.")
    insp.add_argument("--json", action="store_true", help="Output as JSON.")

    dr = sub.add_parser("drop", help="Drop partitions matching a filter.")
    dr.add_argument("path", help="Collection URL or filesystem path.")
    dr.add_argument(
        "--filter", required=True, help="Partition filter expression."
    )
    dr.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )

    return p


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Run the ``zcollection`` CLI and return its exit code."""
    args = _build_parser().parse_args(argv)
    out = out or sys.stdout
    stdin = stdin or sys.stdin
    try:
        if args.cmd == "ls":
            return _cmd_ls(args, out)
        if args.cmd == "inspect":
            return _cmd_inspect(args, out)
        if args.cmd == "drop":
            return _cmd_drop(args, out, stdin)
    except (CollectionNotFoundError, StoreError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return 1
