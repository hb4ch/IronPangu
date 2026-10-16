# Prepared device execution — 2026-10-16

The native decoder now uses a generic per-rank `prepared::Context`. It owns one
ACL stream, a reusable device-buffer pool, persistent tensor/view descriptors,
workspace, pinned control/output staging and qualified repeatable invocations.
The validated DSL still determines graph topology, weights and dispatch; the
context contains no model names, dimensions or layer patterns.

## Ordering and ownership

A temporary device lease is returned to its size class after its last operation
has been **enqueued**, not after a host wait. Reuse is safe because every reader,
writer, D2D copy, memset and collective is ordered on that same owning stream.
The physical allocation remains alive until context teardown. Multiple streams
or concurrent requests cannot share this pool; cross-stream use is rejected by
the context contract. Future overlap needs explicit dependency events and reuse
proofs, rather than relaxing this rule.

Descriptors are cached by allocation address, shape, dtype and transpose layout.
Invocation keys include the preparation function, descriptor identities and all
non-output arguments. Inputs change on device while bindings stay valid. The
qualified repeatable set is Cast, Add, Sub, Mul and Matmul. Retention occurs
immediately after the successful workspace query using
`aclSetAclOpExecutorRepeatable`; retention failure is an error, not a hidden
fallback. DynamicQuant is known to reject repeatability in this CANN version.
Other operators conservatively keep explicit one-shot preparation. Each named
operator's preparation/replay counts and policy are included in the runtime audit.

Workspace grows only when a larger requirement appears. Previous allocations
remain in the pool, so queued users cannot observe freed storage. Compiled kernel
ABI invocations remain single-enqueue; their metadata is retained until output
completion. No ABI repeatability is invented for a foreign kernel.

Control uploads copy caller data into pinned staging slots, then enqueue H2D on
the owning stream. Slots are immutable until the request-output completion and
can then be reused. A temporary host vector may therefore die immediately after
upload. Activation values never enter this staging path: the existing transfer
roles and rejection guards still apply.

## Completion boundaries

Per-operator, router, compiled SwiGLU and collective fences are removed. Output
argmax, diagnostics and token consensus stay on device. Only the previously
allowed 64 bytes/rank of consensus IDs and 80 extra bytes of leader diagnostics
are copied asynchronously to pinned host output storage. **One owning-stream
completion per emitted token** makes those outputs readable and releases retained
invocation metadata/staging slots. Teardown also completes that stream before
resource destruction. There is no device-global synchronization call, rank-wide
host barrier added to normal execution, or implicit activation fallback.

One-time immutable parameter transformations still use their explicit parameter
preparation boundary during initialization. They are not activation offloading.

## Inspect and verify

`INFERFABRIC_PREPARED_AUDIT_DIR` writes one JSON per rank with device allocation
counts/bytes, lease reuse, descriptor count, invocation preparation/replay counts,
output boundary counts, and allocations/reuses/completions for every token position.
Pool bytes include retained expert banks and decoder state allocated through this
context; they are not total HBM use or just live activations. Original checkpoint
storage and vendor-internal allocations are outside that number.

`benchmark-native-tpot.py --resident-audit` checks full output records, all rank
transfer guards, zero decoder allocations and zero internal completion boundaries
at every generated-token position, plus exactly one output boundary per output.
`prepared_execution_probe` tests 128 queued iterations of temporary destruction,
buffer reuse, repeatable invocation reuse and pinned staging with no intermediate
fence, plus cross-stream/double-release/implicit-upload rejection. The existing
resident tensor probe checks ownership, rounding and transfer rejection as well.

Physical-plan annotations and generated viewers describe the new ordering,
allocation-growth and output-boundary contracts. The physical plan remains an
inspection artifact: this milestone prepares and caches backend invocations at
runtime. It does **not** yet execute frozen physical arena offsets or replace
one-shot vendor preparation with full-model ACLgraph. Pool/descriptor growth is
possible for unseen shapes or liveness patterns; the qualified fixed-shape decode
must show zero device allocation after prefill. Capture, statically bounded arenas,
collective reduction and sparse expert dispatch remain separate milestones.

[Measured NPU qualification and TPOT](validation/2026-10-16-prepared-execution/README.md)

## Full graph and shared views (2026-10-16)

The subsequent graph milestone adds owning-stream capture/replay with stable
control buffers. The device-view milestone retains BF16 operator output storage,
shares pool leases across returned values and contiguous slices, and uses those
operands directly in matmul/normalization/collectives. FP32 math boundaries remain
explicit. A view keeps its allocation alive until the final alias is released;
recycling remains ordered on the owning stream. No activation crosses the host.
The generated physical metadata exposes precision and view layout, but does not
claim a frozen arena or exact total HBM.

[Full graph evidence](validation/2026-10-16-native-graph/README.md) ·
[Shared device views: 63.67 ms/token](validation/2026-10-16-device-views/README.md)

The [sparse grouped-expert milestone](validation/2026-10-16-sparse-experts/README.md)
replaces fixed foreign-route compute and weight gathers with device-ordered sparse
group lists and direct bank reads. Its numerical contract is explicit: full greedy
sequences match the pinned reference, while grouped dequantization changes some
logits. Graph replay matches its independently checked eager output records exactly.

[Inference mHC and vector kernels](validation/2026-10-16-vector-kernels/README.md)
reduce the qualified graph baseline to 32.39 ms/token. Fused FP32 rotary, one-pass
packing and vector weighted reduction avoid thousands of small tasks, while the
mHC inference-output contract enables vendor small-token tiling. These kernel
numerical contracts are recorded in generated physical metadata.
