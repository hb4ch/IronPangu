# Batched serving prefix reuse — 2026-10-18

The Rust frontend now derives prospective isolated-chunk arithmetic identity
with the same batch-identity constructor used after native execution. Previously
it predicted a serial identity, so snapshots from continuous batching could not
hit. Token contents, position, peer geometry and member identity remain part of
compatibility; physical resident-slot assignment does not. Incompatible mixed
batch histories conservatively miss rather than silently relaxing numerics.

The HTTP gate uses `device-zero-apc8k-frozen-plan`: two active requests, one
snapshot slot, 4096-token budget, TP8/EP8 and 65,536-token capacity. Snapshot
boundaries align with both page size and complete uncaptured chunks. The checker
now accounts for this alignment instead of expecting a page-only boundary.

| 8176-token request | Cached tokens | TTFT (s) |
| --- | ---: | ---: |
| Cold, salt A | 0 | 5.213 |
| Warm, salt A | 4096 | 2.170 |
| Cold, salt B | 0 | 4.267 |
| Warm, salt B | 4096 | 2.283 |
| Salt A after eviction | 0 | 3.994 |

All outputs matched the independent reference through EOS. A separate source
request of 8176 tokens and target of 16,368 tokens share 8137 initial tokens but
have different suffixes. The target restored the source's 4096-token snapshot
and matched its own reference output (TTFT 6.416 s). This verifies shared-prefix
continuation, not just repeated identical prompts. It is not a cold/warm latency
comparison for the longer target.

Mixed long/short requests, queued isolation, cancellation, disconnect recovery,
context bounds and graceful shutdown also passed. Recorded totals: 42 admissions,
eight mixed ticks, three hits, 12,288 restored tokens and eight publications.
All snapshot entries and shared pages were retired at shutdown. Arithmetic
histories accounted for all 82,315 newly executed scheduler input tokens.

Each rank recorded 2264 explicit D2D state-copy operations totalling 2,194,315,264
bytes across the run. The latest bounded copy plan for rank 0 is included; it is
not a log of every copy. All eight ranks retained the 40,492,359,704-byte arena
and 159,221,248-byte workspace, with no growth, execution allocation, internal
completion boundary or activation-host transfer. State-copy completion uses its
existing owning-stream boundary; no global fence was added.

Local checks: 10 native Rust tests, 30 frontend tests and nine HTTP checker tests
passed. One separate NPU backpressure test remains ignored in local testing.
Rust was built/cross-compiled locally in WSL; inference ran in Docker. HTTP runner
exited 0, HCCL port 63600, remote output `device-zero-apc-http` under
`/data/p00603624/inferfabric/openpangu92`. The frontend binary was
`/data/p00603624/inferfabric/inferfabric-server-apc-device-zero`.

These tests qualify one snapshot slot and chunk-compatible histories. Larger
snapshot budgets, 64K warm-prefix restoration, sustained load and fault recovery
remain additional gates; this is not a claim of production readiness.
