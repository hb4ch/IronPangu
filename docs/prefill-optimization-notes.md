# Prefill investigation — 2026-10-18

Priority is full-model uncaptured 4K prefill, then mixed serving latency. Keep
decode ACLGraph and device residency. Use small reference fixtures to establish
an effect before repeating the 64K qualification.

## Evidence and interpretation

The serving capture and HOST/AIV comparison are recorded under
`docs/validation/2026-10-18-prefill-profile` and
`docs/validation/2026-10-18-communication-ab`. Inclusive AllReduce and notify
wait durations dominate the timeline, but those include rank readiness. An AIV
environment change alone did not fix latency. This does not establish a network
bandwidth bottleneck.

Packed expert outputs previously constructed one ACL view per routed row even
when consumers only used the packed allocation. Preserve row geometry and make
views only for operations that actually request rows. This removes redundant
descriptor lookup and ownership work without changing the physical kernels,
copies or device allocation sizes. Qualification must measure whether it helps;
source-level work reduction alone is not a performance claim.

## Next candidate: explicit device zero

The capture records 712 `aclrtMemsetAsync` calls across eight ranks, with about
15.29 seconds of summed host API time (overlapping across threads). Source in
the local CANN checkout, `runtime/src/runtime/core/src/launch/memory_common.cc`,
implements a `DevMemSetAsyncByMemcpy` alternative that allocates and initializes
a host buffer, transfers it, and replicates it using D2D copies. This is
consistent with the observed overhead, but does not prove that the installed
runtime selected that branch.

Evaluate an explicit device-zero operator for `expert_zero`, whose large packed
output is repeatedly cleared. The installed `aclnnop/aclnn_zero.h` exports
`aclnnInplaceZeroGetWorkspaceSize` and `aclnnInplaceZero`. Verify the supported
storage dtype and inspect actual kernel/copy behavior; its API description
includes ZerosLike/ViewCopy and cannot be assumed to be a single kernel.

Implement selection in the physical plan and its viewer description. Do not
silently replace a declared memset with a different execution sequence. Reserve
any workspace before freezing the arena, preserve stream ordering and graph
capture, and verify EOS-scoped output, allocation/transfer audits, actual
prefill and decode timing. If it helps, qualify mixed requests and long contexts
before changing the serving default. Do not attribute reductions in collective
waits to communication improvements when rank readiness changed.

## Current completion-scoped backend

The [new 4K capture](validation/2026-10-18-current-prefill-profile/README.md) passes all 39 fixture tokens with the current frozen backend. Its 3.01-second profiled batch still shows large inclusive HCCL/notify intervals after device zero and metadata-lifetime fixes. Inspect actual collective implementation selection and host expansion before further kernel changes; waiting intervals alone do not prove bandwidth saturation. Keep the 4K uncaptured quantum and validate a targeted change on the smaller fixture before repeating 64K.

## Next implementation: explicit output-row demand

The current capture's `collectives.json` reports, on every rank, 92 FP32
AllReduces of 10,485,760 elements (40 MiB input each), and one BF16 AllGather of
77,594,624 elements (148 MiB input). The captured middle prefill has no sampling
rows. Source confirms that `execute_token` nevertheless executes every output
node, including the vocabulary projection and gather, before `emit_rows` sees an
empty row list. This is a concrete output-demand gap, independent of the remaining
AllReduce bottleneck. Removing it alone is not expected to remove the whole delay.

Implement this as a generic compiler/runtime contract, not a model-name check or
an executor-only skip:

1. Add an explicit request-output row domain to IR. The scheduler supplies packed
   sample-row indices in deterministic member order. Full-token state updates
   remain mandatory effect roots even when no output rows are requested.
2. Identify the row-independent output suffix using declared operator effects and
   row semantics. Insert a physical row-selection operation at its boundary and
   annotate downstream shapes with the sampled-row extent. Reject unsafe movement
   across attention, recurrence, routing, reductions across tokens or state writes.
   Default unknown operators to non-movable. Do not change reduction dtype/order.
3. Zero sample rows skip only the proven output suffix. Final prefill and mixed
   batches evaluate that suffix for precisely the requested rows. Return a compact
   logits row mapping; sampling must not use the original packed token offsets.
4. Reserve control buffers, selected hidden rows and output workspaces before arena
   binding. Preparation must cover sample counts 0 through resident capacity and
   mixed tails. Decode graph row counts and pointers remain stable; long prefill
   stays uncaptured. The physical plan/viewer must show selection, row extents and
   the skipped output collective, with no hidden casts/copies.
5. Test compiler legality with stateful/unknown-op rejection, then native empty,
   final and mixed sample-row cases against independent full-model references.
   Verify intermediate prefill has no vocabulary collective using the audit tool;
   measure the same uninstrumented 8K fixture before extending to the 64K gate.

The local HCCL source's A3 selector limits ordinary AIV by per-rank payload (1 MiB
in its op-base path). A 40 MiB reduction over eight ranks exceeds that threshold.
This is a possible explanation for the earlier environment-only AIV result, not
proof of the installed binary's selection. Its profiler label is RING-RING-NHR;
that label does not identify an executor class. Do not silently force a newer
source's AIV_ONLY enum into the installed communicator ABI.

### Implemented legality analysis (not enabled in execution yet)

`inferfabric_compiler::output_rows::plan` now proves a conservative contiguous output suffix and returns its source, bounds and explicit request-sample row domain. Unknown/stateful operations stay in the full-token prefix; nonterminal outputs are not transformed. Rank-major gather plus token-layout conversion must move together. Three local legality tests and the real 92B learned-C8 DSL lowering test pass. The current binary and executor remain unchanged: serialization, viewer presentation, compact runtime row mapping, preparation coverage and NPU qualification are still required before this optimization is enabled.

### Executable candidate: IFTOKN08

Compilation with `INFERFABRIC_EXPERIMENTAL_OUTPUT_ROWS=1` now emits an explicit
`output_rows` contract in the physical program and a versioned IFTOKN08 header
(source ID/width, first and last suffix instructions). Both compiler validation
and native loading reject unsafe suffixes. This is an opt-in candidate undergoing
NPU qualification; existing default bundles still use the qualified format.

The uncaptured executor performs declared ordered device row copies before the
suffix, switches its tensor extent to requested sample rows, and returns compact
logits in member order. Empty sample lists finish the full state-updating prefix
and skip only that suffix. Pure decode keeps fixed sequence rows and existing
graphs. Quantized compound inputs are not treated as plain row tensors: the
planner stops before an unsafe boundary rather than reinterpreting their storage.

Preparation covers zero through resident-capacity sampled rows before freezing
workspace/arena. Viewer recipe details expose the selection and dynamic row axis,
including the rank-major gather/layout pair. Local compiler tests, cross-build,
native Docker build and the viewer regression pass. The full NPU discovery,
fixed-workspace and frozen-arena qualification is still required before acceptance.

The [three-stage native qualification](validation/2026-10-18-output-rows/README.md) now passes 39/39 tokens per stage with the executable candidate. Frozen arena reservation falls by 2,482,421,272 bytes per rank. 64K HTTP/APC and a post-change collective capture remain pending; the compilation opt-in stays in place.

## Qualified communication-domain candidate

The [40 MiB collective-domain test](validation/2026-10-18-collective-domains/README.md) passes synthetic reduction correctness and interleaved HOST decode graph replay with an AI_CPU bulk domain. Median worst-rank call time is 0.899 ms versus 35.983 ms for HOST in this diagnostic. This justifies explicit per-variant communicator planning and full-model qualification; it is not a model speedup claim. Keep the process-global HOST mode fixed, create domains before preparation, and expose their memory/selection in the physical plan.


### Integration priority: bulk communication on the real serving path (2026-10-18)

The next gate is the real packed/mixed serving path, not another synthetic
collective sweep. The candidate compiler now emits `IFTOKN09`, assigning bulk
AllReduce to a separately initialized AI_CPU communicator while retaining HOST
for decode graphs, AllGather and sampling consensus. Both domains use the same
owning compute stream. No new stream/global fence or activation staging is added.
The assignment survives compact output-row selection and appears in the generic
plan viewer. The buffer configuration is explicit; HCCL allocations remain
outside the activation arena and must be included in measured HBM occupancy.

Qualification order is full model discovery/fixed/frozen output and resource
checks, then long-context HTTP mixed scheduling and prefix-cache recovery. The
synthetic result alone does not authorize making this the default. Primary
acceptance metrics are uncached 4K forward time, long-prompt TTFT, short-request
latency under mixed work, pure decode TPOT, correct first-EOS output and bounded
memory. Continuous batching and bulk prefill are prerequisites for production;
other optimization work must not displace these gates.
