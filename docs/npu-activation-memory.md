# Activation memory management

## Milestone 1: stable operator workspace pool

Prepared operators query their workspace requirement without allocating it. Before the first eager execution or capture, the root sequence binds its unbound operators to a session-owned block sized to the largest requirement. Subsequent sequences reuse a suitable existing block or allocate a new stable block; previously bound addresses never change. Captured graphs are destroyed before the pool. Stream-serialized execution and synchronous public execution/replay boundaries are required for reuse.

Request RNG also uses the pool after the preceding replay fence and completes before the next replay. `INFERFABRIC_DEDICATED_WORKSPACES=1` retains the original per-operator allocator for comparisons. `workspace_bytes` reports physical owned memory, including pooled blocks, rather than summing aliased logical requirements.

Qualified in Docker on Ascend A3, CANN 9.1.0, TP2, Qwen3.5-2B, context 2048, four sequences and 16 scheduled tokens. Native C++ builds with warnings treated as errors. Both ranks passed eager/graph bitwise comparison, full-context and inactive-lane state checks. The Rust frontend native, sampling and batching smoke tests passed, including 12 concurrent requests, isolated-baseline agreement, seeded penalties, mixed prefill/decode, cancellation and slot reuse.

| Per-rank measurement | Dedicated baseline | Shared workspace |
|---|---:|---:|
| Workspace bytes | 1,394,039,040 | 33,555,968 |
| Observed HBM growth, rank 0 | 5,788,831,744 | 4,260,388,864 |

Profiles are in `validation/2026-09-07-activation-memory/workspace-tp2-rank{0,1}.json`. Observed HBM includes CANN/runtime overhead and is a watermark rather than an exact allocation trace. Tensor activation reuse is the next milestone.

## Milestone 2: activation lifetime reuse

The Rust planner consumes the lowered layer order. Hidden values have inclusive live intervals spanning their producing and consuming operations, including the residual path. A checked, 512-byte-aligned arena places non-overlapping intervals at reusable offsets. For 24 layers this reduces 26 hidden/normalized allocations to two slots. Native buffer views never own or free the arena parent.

The C++ composite boundary is a conservative lifetime boundary: all scratch within one composite gets distinct slots; sequential composites reuse those slots. TP local matmul outputs and gather buffers stay live through HCCL and its following stream-ordered copies. Layer intermediates use a separate pool and remain live until the entire layer finishes. Recurrent/conv backups use another pool and remain live through the inactive-lane restore. Immutable uploaded constants, weights, persistent KV/recurrent/conv state, request controls, and logits retain independent storage.

Pools reuse suitable blocks and retain older versions if a later prepared shape needs a larger block. No captured address moves. All graphs and repeatable executors are destroyed before pool storage, after the session stream fence. Each rank has its own pools; the serving worker serializes operations and graph replays. Adding overlapping execution streams requires a corresponding change to pool ownership or event dependencies.

`INFERFABRIC_DEDICATED_ACTIVATIONS=1` disables tensor reuse in both Rust and C++. Set it together with `INFERFABRIC_DEDICATED_WORKSPACES=1` for the original allocation strategy. These are startup settings inherited by every rank. Allocation details in every profile stage distinguish physical scratch, layer and snapshot pools, owned temporaries, logical scratch/workspace demand, owned buffers and non-owning views. Profile fingerprints cover startup logits, every KV/recurrent/conv state region, and full-context logits.

The planner is conservative rather than a minimum-memory solver: it does not alias values within a C++ composite or within a layer. This keeps internal ACLNN and HCCL lifetimes explicit while removing cross-operation and cross-layer retention. Fixed serving shapes bound request RNG storage; request setup reuses workspace and fences before replay.

Reproduce the HTTP comparison inside Docker with `scripts/qualify-activation-memory.py --output dedicated.json --rounds 3` against a server started with both dedicated settings, then restart with pooling and run `scripts/qualify-activation-memory.py --reference dedicated.json --output pooled.json --rounds 5`. The harness only sends HTTP requests to the Rust frontend. It checks greedy/seeded/penalty outputs, isolated versus mixed batches, and per-device process HBM stability after warmup. Timing includes queueing and long prefills; it is a regression observation, not an isolated decode benchmark.

### Qualification results

Both TP2 ranks produced identical dedicated/pooled SHA-256 fingerprints for startup logits, full-context logits and all 48 persistent state regions. The native sampler CPU reference covered 12 distributions (maximum probability error 1.49e-8), 2,048 categorical draws and seeded graph replay. TP matmul matched CPU/eager/graph for replicated and sharded weights with three changing inputs. Frontend smoke tests covered chat/completions, SSE, seeded sampling, penalties, mixed prefill/decode, cancellation and slot reuse. The larger TP1 startup case, context 8192 with eight slots, also passed.

| Measurement | Dedicated | Pooled |
|---|---:|---:|
| TP2 rank 0 observed startup HBM growth | 5,788,581,888 | 3,531,329,536 |
| TP2 per-rank workspace bytes | 1,394,039,040 | 33,555,968 |
| TP2 per-rank temporary bytes | 598,634,904 | 56,263,112 |
| TP1 context 8192, eight slots, observed growth | 9,231,245,312 (previous qualification) | 5,469,540,352 |

The TP2 pooled temporary breakdown is 51,911,232 bytes of composite scratch, 4,341,760 bytes of state snapshots and 10,120 bytes of owned temporaries/constants. The separate layer pool is 709,248 bytes. Weights and persistent state sizes are unchanged. Rust workspace tests (34), frontend tests (6), Clippy and Docker C++ warnings-as-errors builds passed.

The direct HTTP corpus matched the dedicated reference exactly, including output usage and finish reasons. Five pooled rounds (60 mixed requests, after 12 isolated requests) held per-rank process HBM at 3487/3481 MiB in every reading; the dedicated run held 5643/5637 MiB. This is zero observed growth at `npu-smi` MiB resolution. Median/p95 queued request latency was 5.38/16.08 seconds pooled versus 6.31/17.73 seconds dedicated. The workloads include long prefills and these short runs do not establish a general throughput speedup. See `pooled-corpus.json` and `dedicated-corpus.json` in the validation directory.
