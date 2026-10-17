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
