"""CLI entry point: ``python -m zcollection.benches all --store <url> --out v3.json``."""

import argparse
import sys

from .harness import BenchSpec, compare, dump_json, run_suite


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI and return its exit code."""
    p = argparse.ArgumentParser(prog="zcollection.benches")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("all", help="Run the full benchmark suite")
    run.add_argument(
        "--store", required=True, help="Store URL (e.g. file://, s3://...)"
    )
    run.add_argument("--out", required=True, help="Output JSON path")
    run.add_argument("--n-partitions", type=int, default=12)
    run.add_argument("--rows-per-partition", type=int, default=50_000)
    run.add_argument("--width", type=int, default=240)
    run.add_argument("--profile", default="cloud-balanced")

    cmp = sub.add_parser(
        "compare", help="Compare a result file against a baseline"
    )
    cmp.add_argument("current", help="JSON file from a prior 'all' run")
    cmp.add_argument("--baseline", required=True, help="Baseline JSON path")

    args = p.parse_args(argv)

    if args.cmd == "all":
        spec = BenchSpec(
            n_partitions=args.n_partitions,
            rows_per_partition=args.rows_per_partition,
            width=args.width,
            profile=args.profile,
        )
        results = run_suite(args.store, spec)
        dump_json(results, args.out)
        for r in results:
            sys.stdout.write(
                f"{r.name:32s} {r.seconds:8.3f}s  counts={r.counts}\n"
            )
        return 0

    if args.cmd == "compare":
        import json
        from pathlib import Path

        from .harness import BenchResult

        raw = json.loads(Path(args.current).read_text())
        current = [BenchResult(**item) for item in raw]
        ratios = compare(current, args.baseline)
        for name, ratio in ratios.items():
            sys.stdout.write(f"{name:32s} {ratio:6.2f}x\n")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
