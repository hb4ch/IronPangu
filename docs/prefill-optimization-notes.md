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
