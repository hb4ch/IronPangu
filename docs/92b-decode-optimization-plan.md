# 92B decode profiling and optimization plan — 2026-10-16

The native device-resident TP8/EP8 path is dominated by host orchestration,
collective submission and synchronization. Faster GEMMs alone will not fix the
current approximately two-second token latency. This document records measured
bottlenecks and the implementation order; the optimizations below are not yet
implemented or assigned speculative speedups.

## Measurement

Hardware: eight NPUs on `7.156.99.58`, Docker `inferfabric-92b`, checkpoint
`/models/92B`, compact reusable 92B DSL, 46 layers, batch one, context capacity
512, greedy decode, serial eager execution, `HCCL_OP_EXPANSION_MODE=HOST`.
No other NPU processes were observed before profiling. Rust tooling remains
locally built; native C++ and NPU execution run inside the remote container.

The prior unprofiled output-to-output measurement was **2,046.86 ms/token**.
Lightweight host timers now measure two full decoder iterations after the first
output in each of three requests. These are early decode steps, not a claim that
all lazy allocations have reached steady state. All eight ranks are recorded.
The timed decoder excludes output sampling, diagnostics and consensus; the
vendor capture includes the two following output boundaries. Full generation
continues beyond the capture: all 33 records, including diagnostic scores, match
the qualified resident reference exactly. Every rank reports zero activation
host transfers and zero rejected transfers.

| Case | Timed steps | Mean across ranks, ms/step |
|---|---:|---:|
| Arithmetic | 2 | 1,983.90 |
| Geography | 2 | 2,110.24 |
| Chinese | 2 | 2,090.71 |
| Equal-weight mean | 6 | **2,061.62** |

This small sample diagnoses the current short-context path. It is not a serving
benchmark, long-context result, or concurrency qualification. Rank averages are
not a critical-path decomposition.

## Actual bottlenecks

Logical operation timings include their internal launches, copies and waits.
Values below average the six steps and eight ranks.

| Logical operation | ms/token | Fraction of decode |
|---|---:|---:|
| All-gather nodes (277/token) | 952.26 | 46.2% |
| Expert gate/up projections | 248.75 | 12.1% |
| MLA query construction | 228.90 | 11.1% |
| Expert SwiGLU/down | 151.09 | 7.3% |
| All-reduce nodes (44/token) | 95.21 | 4.6% |
| MLA value construction | 69.09 | 3.4% |

A separate API-level view explains much of that wall time. **Do not add this
view to the logical-operation table.** API spans include blocking and runtime
work; they are not device execution durations.

| API activity | Calls/rank/token | Mean ms/rank/token |
|---|---:|---:|
| HcclAllGather submission | 277 | 751.87 |
| Stream synchronization | 15,852 | 501.60 |
| Device-to-device copies | 12,711 | 365.18 |
| Kernel launch wrappers | 15,441 | 150.56 |
| Device allocation | 28,222 | 87.23 |
| Device free | 28,222 | 74.68 |
| Explicit control-input H2D copies | 874 | 13.23 |
| HcclAllReduce submission | 44 | 13.09 |

The H2D calls are classified control inputs, not activation staging. Parameter
preparation may transfer immutable weights during initialization. Neither is a
permission for activation fallback.

The separate CANN capture provides device-level evidence:

- Rank 0 records **65.79 ms of compute-task interval union per token**. Across
  all eight devices this is 63.38–66.02 ms. Mixed-kernel child tasks overlap their
  parent and are not added again.
- QuantBatchMatmulV3: 1,154 calls/token, 15.21 ms device time. MatMulV2: 821 calls,
  5.23 ms. Cast: 6,599 calls, 9.80 ms. GatherV3: 2,289 calls, 12.44 ms.
- HCCL records 278 all-gathers/token including output consensus, and 44
  all-reduces. Their inclusive spans are 929.43 and 64.53 ms/token respectively.
- HCCL copy/reduce tasks move approximately 5.76 MB/rank/token and sum to only
  **2.28 ms/token** on rank 0. Notify-wait spans are large and overlap streams.
  This supports launch, synchronization and rank-arrival overhead as the problem,
  rather than saturation of link bandwidth. It does not isolate each wait's cause.
- Sparse attention itself is only 2.40 ms/token of recorded device task time.
  Sampling/top-k is also small in this capture.

Vendor tracing raises the mean host decode time to **3,867.12 ms/token**, about
85% above the lightweight Chinese run. Its trace gaps are not a utilization
counter or an attainable latency bound. Use lightweight/unprofiled wall times
for performance decisions; use the trace for task counts, ordering and device
work. Do not subtract device times from a different run's host times.

## Ordered implementation milestones

All stages retain the planner's hard device-residency legality rule. Host
activation candidates never enter cost search. Route IDs, route counts, scores,
intermediates and KV remain device-resident. Only named control inputs,
initialization parameter preparation and request outputs may cross the boundary.

### 1. Prepare invocations and reuse device memory

Implement a generic prepared-invocation executor consuming physical value
lifetimes, layouts and workspace requirements. Allocate persistent activation
arenas and operator scratch before decode; bind outputs directly to planned
storage. Replace materialized slice/concat operations with checked views where
layout permits. Retain BF16 between compatible operations, while preserving
explicit FP32 accumulation and required rounding boundaries.

Stop synchronizing after every operation. Same-stream order supplies dependencies;
use events for actual cross-stream dependencies and safe buffer reuse. Executors,
descriptors, workspaces and temporary storage must outlive queued work. Confirm
which vendor executor objects are reusable rather than assuming all are.

Acceptance: no per-token device allocation/free after preparation for supported
shapes; documented remaining copies and fences; identical complete output records;
zero activation host transfers; lifetime/alias tests and repeated request resets.
Expose arena offsets, live ranges, views, workspace ownership and synchronization
reasons in the generic physical-plan viewer. Measure the new TPOT before stage 2.

### 2. Capture stable device execution regions, including collectives

Once pointers and lifetimes are stable, qualify ACLgraph capture/replay for the
prepared schedule on all eight ranks. Stable collective order and dynamic token,
position and cache metadata must be explicit inputs. The 752 ms/token spent
inside all-gather submission makes repeated host expansion a priority.

Use qualified regions if a vendor operator cannot be captured. Region boundaries
may add device events; they must not move activations to the host. Keep the eager
device path as a correctness oracle. Existing component capture probes do not
qualify full-model capture.

Acceptance: graph/eager output equivalence over multiple positions, requests and
cache resets; identical rank collective order; capture failure is explicit;
reduced host submission/fence counts and measured TPOT improvement. Show capture
regions and uncapturable reasons in the viewer.

### 3. Reduce communication through distribution-aware planning

Current TPLinear always gathers its output. First implement transformations that
preserve existing arithmetic:

- Keep q_b output head-local: each rank needs six of the 48 heads, yet the current
  path gathers all 9,216 values and then selects its local heads. A verified local
  MLA-query input contract can remove 46 gathers/token.
- Pack independent gate/up projection results into one collective, unpacking to
  the same values. Across dense/shared FFNs this can remove another 46 gathers.

The candidate count is **277 → 185 model all-gathers/token**, conditional on
layout and dependency validation. Cost the packing/copy overhead too; a count
reduction alone does not prove a speedup.

Later consider sharded SwiGLU and input-column down/output projections followed
by reductions. Dynamic quantization currently depends on the complete activation
range: blindly quantizing independent shards changes scales and output. Preserve
global scales explicitly or qualify a new numerical contract. Reduction-order
changes also require numerical validation. Record placement transitions,
collective sequence and chosen/rejected alternatives in the viewer.

### 4. Execute only owned routed experts

The current resident path computes eight route slots on every rank and masks
foreign-owner contributions. It performs 64 rank-slot computations for eight
globally selected experts per MoE layer, and gathers expert weights into temporary
operands. Across 44 MoE layers this produces 1,056 expert projection calls per
rank/token (three projections per slot).

Add a generic device-routed expert dispatch contract with device counts, offsets,
weight descriptors and grouped execution. Do not read routing decisions back to
the host to choose launches. Qualify the backend's device-count support; otherwise
use a bounded, device-predicated execution design. Expected balanced work is one
owned selected expert per rank, but skew and empty ranks must be supported.

Preserve quantization scales, route-slot reduction ordering and FP32 accumulation.
Remove duplicate packed banks only when direct access is qualified. Acceptance:
correct empty/skewed routing, bounded capacity, unchanged output, no route D2H,
and measured kernel/copy reductions. Do not promise an eightfold TPOT speedup.

### 5. Batch MLA suboperations and fuse small operator chains

MLA query construction costs about 229 ms/token on the host despite modest device
math. Batch the per-head absorbed projections, RoPE and layout operations where
legal. Reduce Cast/Gather/materialization launches through dtype legalization,
views and qualified fusion. Preserve original BF16 rounding and FP32 operations;
fast-math or changed quantization needs separate numerical acceptance.

Defer isolated attention/sampler tuning until the new profile demonstrates that
it matters. Kernel choices remain backend contracts, not model-name conditionals.

### 6. Feed measured costs into planning and broaden qualification

Persist hardware, CANN, binary/source identity, shape, dtype, placement, batch,
context, phase and measurement method with costs. Keep host submission, device
execution, transfer and synchronization costs separate. Model launch amortization
and rank critical paths; do not sum overlapping trace spans. These aggregate
samples establish priorities, not a complete per-shape optimizer cost database.

After each optimization, re-run unprofiled TPOT and the same capture, then extend
to longer outputs/contexts, repeated requests, skewed routes and concurrent loads.
Keep exact output for scheduling/storage-only changes. Arithmetic rewrites need
an expanded reference corpus and documented score/error bounds, with correct
end-to-end text. Recheck transfer rejection, memory bounds, graph/eager equivalence
and rank failure handling. Commit and push each qualified milestone.

## Evidence and current limitations

[Raw host timers, audits, output records and device summary](validation/2026-10-16-92b-profile/README.md)
contain the reproducible measurements. No optimization in this plan has been
credited with a speedup. The current physical plan is inspectable planning IR;
implementing its prepared execution contract remains work, not an already-frozen
production execution engine.
