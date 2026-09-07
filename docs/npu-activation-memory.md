# Activation memory management

## Milestone 1: stable operator workspace pool

Prepared operators query their workspace requirement without allocating it. Before the first eager execution or capture, the root sequence binds its unbound operators to a session-owned block sized to the largest requirement. Subsequent sequences reuse a suitable existing block or allocate a new stable block; previously bound addresses never change. Captured graphs are destroyed before the pool. Stream-serialized execution and synchronous public execution/replay boundaries are required for reuse.

Request RNG also uses the pool after the preceding replay fence and completes before the next replay. `PANGU_DEDICATED_WORKSPACES=1` retains the original per-operator allocator for comparisons. `workspace_bytes` reports physical owned memory, including pooled blocks, rather than summing aliased logical requirements.

Qualified in Docker on Ascend A3, CANN 9.1.0, TP2, Qwen3.5-2B, context 2048, four sequences and 16 scheduled tokens. Native C++ builds with warnings treated as errors. Both ranks passed eager/graph bitwise comparison, full-context and inactive-lane state checks. The Rust frontend native, sampling and batching smoke tests passed, including 12 concurrent requests, isolated-baseline agreement, seeded penalties, mixed prefill/decode, cancellation and slot reuse.

| Per-rank measurement | Dedicated baseline | Shared workspace |
|---|---:|---:|
| Workspace bytes | 1,394,039,040 | 33,555,968 |
| Observed HBM growth, rank 0 | 5,788,831,744 | 4,260,388,864 |

Profiles are in `validation/2026-09-07-activation-memory/workspace-tp2-rank{0,1}.json`. Observed HBM includes CANN/runtime overhead and is a watermark rather than an exact allocation trace. Tensor activation reuse is the next milestone.
