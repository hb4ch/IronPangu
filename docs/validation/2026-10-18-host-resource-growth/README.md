# Mixed-load host resource regression — 2026-10-18

**Failed resource gate.** The unchanged qualified 92B backend completed 24
four-client waves containing 8176-token retrieval prompts and short requests.
All 120 completed wave/recovery requests matched independent references through
EOS, with checked token usage. Each wave also deliberately disconnected one
stream. The first completed wave's RSS was 7,521,230,848 bytes; the last was
8,311,930,880 bytes. Maximum growth, 790,700,032 bytes (754 MiB), exceeded the
128 MiB tolerance selected before the run. Threads remained 631 and file
descriptors 76. This is evidence of excessive process memory growth, not yet
proof of a particular leak or allocator cause.

The runner exited 1 after persisting raw per-wave samples and the negative
gate result. Graceful cleanup completed: 171 admissions, shared-page occupancy
zero, native workers closed. Inspection of all eight shutdown audits found
arena 40,492,359,704 bytes per rank, zero workspace growth, zero execution
device allocations, zero internal completion boundaries and zero activation
host transfers. Device-memory invariants alone therefore miss this regression.
The runner did not emit its full success report; do not classify this run as a
passing serving qualification.

Rank 0 retained 31,008 descriptors and 1,051 invocations at shutdown, compared
with 29,589 and 1,049 in the earlier 12-wave run. Transient invocations were
zero at shutdown. These endpoint counts are not sufficient to attribute RSS
growth; distinguish descriptor retention, vendor allocations and libc retention
before changing lifetimes that decode graphs depend on.

Reproduce with `qualify-program-http.py`, `--load-rounds 24
--load-concurrency 4 --load-rss-growth-limit-mib 128`, independent short cases
and 8K resident cases, two active sequences, a 4096-token budget,
`--respect-eos --check-arithmetic-history --expect-prefill packed`.
The server is `inferfabric-server-apc-device-zero`, plan
`device-zero-apc8k-frozen-plan`, library `native-build-device-zero`, checkpoint
`/models/92B`. Prefix caching is disabled. Remote evidence is
`/data/p00603624/inferfabric/openpangu92/resource-load-http`, HCCL port 64400.

The next controlled experiment limits glibc to two arenas with
`MALLOC_ARENA_MAX=2`, retaining the same workload, binary, plan and tolerance.
This is a diagnostic experiment, not an adopted production fix.
