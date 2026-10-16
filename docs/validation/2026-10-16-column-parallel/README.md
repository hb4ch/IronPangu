# Column-sharded output projections — 2026-10-16

**25.7072 ms/token**, 38.8997 tokens/s, versus 26.4094 ms/token previously.
The optimized OmniInfer reference is 18.1171 ms/token: a **1.42× gap** remains.
Median is 25.6927 ms and empirical p95 25.9252 ms over 60 intervals, six runs.

The DSL now keeps the two dense FFNs' gate/up outputs local and applies their
down projections with input-column TP slices. Attention output MoME channels
remain local, followed by a column-sharded output projection. Partial outputs
are BF16 and explicit all-reduces accumulate in FP32. Shared experts remain
replicated. The expanded graph contains 1,881 nodes, one gather and 92 reductions,
versus 1,931 nodes, 99 gathers and 44 reductions before this change.

The permanent initialization IR adds optional strided source rows to `LoadSlice`.
Wire opcode 2 appends row width and source stride; opcode-1 artifacts remain
unchanged and accepted. The generic loader validates the full source span,
packs selected rows into bounded staging, and uploads contiguous destination
chunks. It does not load the full matrix, interpret model names, or create a
model-specific loader. Per-output quantization scales remain replicated.

Inspect the [generated plan](execution.html) and the
[48 strided rank-0 loads](strided-load-excerpt.json). The latter is explicitly an
excerpt, not a complete initialization artifact. Physical metadata identifies
local quantization/BF16 partial-output semantics and explicit reduction nodes.

All 33 greedy tokens match the independent OmniInfer reference. Maximum selected
log-probability differences are 0.030400, 0.000859 and 0.186348 for the three
prompts. Local quantization/reduction changes logits; old fixtures are retained.
All six graph runs match 66 complete eager records exactly and pass 48 rank
residency/replay/allocation/completion audits. Activations remain device resident;
no implicit host fallback or global fence is introduced.

Local validation passes 43 compiler, 16 DSL and 3 initialization-IR tests.
Native streaming copy/cast/strided-load tests pass inside Docker, including
source-range rejection and cleanup. All eight bindings and nine negative kernel
contracts pass for both the new and previous plan. Rust cross-build is local.

Same checkpoint, devices 0–7, TP8/EP8, batch one, context 512 and short prompts as
prior milestones. Loading/prefill are excluded. This is an offline decode result,
not a concurrent-serving or long-context performance qualification.
