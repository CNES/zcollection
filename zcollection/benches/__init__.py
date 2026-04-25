"""Benchmark harness for zcollection.

Drives a small, well-defined set of operations (open / query / insert / map)
against any store URL and emits machine-readable results that can be diffed
against a captured baseline. Acceptance targets versus the legacy Zarr v2
codebase:

- ``query_one_partition_full`` ≥ 3x;
- ``insert_one_partition`` ≥ 2x with PUT count ≥ 8x lower;
- ``open_collection_cold`` ≥ 5x;
- ``local_insert_baseline`` ≥ 1.0x (no regression).

Usage::

    python -m zcollection.benches all --store s3://my-bucket/zc-bench --out v3.json
    python -m zcollection.benches compare v3.json --baseline v2-baseline.json

Counter probes (PUT/GET) are best-effort: they require the store to expose a
counting hook. For ObjectStore we wrap the inner obstore client; for local
FS we count file writes/reads via ``os.stat`` deltas.
"""

from .harness import BenchResult, BenchSpec, run_suite
from .probe import CountingProbe

__all__ = ("BenchResult", "BenchSpec", "CountingProbe", "run_suite")
