# Executable storage planning and simpler viewer — 2026-10-16

The compiler now emits `IFTOKN02` rather than executing the logical graph
unchanged. The 92B plan has 1,929 executable instructions, including **228
explicit boundary casts**, 92 AllReduce operations and one AllGather.

The ownership pass eliminates **138 convolution staging copies per token**.
It handles disjoint nested slices, rejects overlapping readers/output escapes,
and inserts a protective copy when exclusive ownership cannot be proven.
Native validation repeats the proof. In this model no protective copies remain
for convolution; other recipe-internal copies still exist.

Selected projection kernels are bound to checkpoint dtype; native validation
rejects mismatched kernels or storage contracts. Identical casts may be shared,
but this 92B graph has zero such common conversions. FP32 addition followed by
BF16 rounding remains explicit. This does not claim that all 228 casts can be
removed without changing arithmetic.

The [viewer](execution.html) starts with modules and shows kernel APIs,
precision/storage, materialization and TP/EP communication in a paged table.
Filters expose casts, collectives/EP and copies; details show source bindings
and dependencies. Rank selection exposes expert ownership. EP uses replicated
routing, owner-local experts and explicit AllReduce of partials. All-to-all and
AGRS are marked unsupported. Vendor recipes and unresolved costs are labeled.

Validation: 64 compiler/DSL tests pass locally. All eight native rank bindings and 18 negative contracts pass in Docker.
All six final full-generation graph runs match the same qualified complete
output fixture: 66 records, including diagnostic scores, and 48 clean rank
audits. Device residency and warmed allocation/completion rules pass.
The fixture was not regenerated or weakened. See `runs.json` and `summary.json`
for final hashes and timings. Final TPOT is **19.9519 ms/token** (median
19.8793 ms, p95 20.8127 ms), versus 20.0132 ms previously. This small change
does not establish parity with the 18.1817 ms OmniInfer reference. Initializations remain byte-identical across all
eight ranks to the paired-projection plan.

This is an executable **storage-boundary milestone**, not completion of the full
kernel-level optimizer. Composite recipes still own internal launches and
workspace; the frozen physical arena, complete recipe expansion and measured
kernel/layout/EP cost search remain unfinished. The design records these stages
in [native physical planning](../../native-physical-planning.md). Activations
remain device-resident; no host fallback or global fence was introduced.

Browser validation passed hierarchy navigation, filters, kernel search, rank
selection and dependency details with no JavaScript errors. Rust was built
locally; native builds and execution were confined to the remote Docker container.
