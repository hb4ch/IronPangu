# Batched MLA heads — 2026-10-16

The full 92B TP8/EP8 graph measures **29.5091 ms/token**, versus 29.9477 ms for
the preceding rank-placement milestone. This modest 1.5% improvement still leaves
a 1.63× gap against the matched OmniInfer 18.1171 ms/token reference.

Query absorption and value expansion each use one BatchMatMul for six local
heads. Query rotary work is batched across those heads, retaining FP32 rotary
arithmetic and BF16 output rounding. Generic bounded strided descriptors expose
the existing parameter storage without host materialization. Descriptor cache
keys include storage extent, dimensions, strides, dtype and pointer. Native
tests reject invalid rank, zero stride and out-of-bounds views and exercise
cache separation; full-model runs validate actual batched computation.

All 33 greedy tokens match the independent reference. Maximum selected-token
log-probability differences are 0.030586, 0.004995 and 0.076362 across the three
prompts. Batched matmul has a distinct numerical contract in physical metadata;
it does not claim bitwise logits against per-head matmul. Graph replay matches
all 66 eager output records exactly, with 48 rank residency/allocation/completion
audits passing. No activation host transfers or global fences were added.

All 42 compiler tests pass locally and the compiler was cross-built locally.
Native compilation, prepared-buffer tests and model execution ran in Docker.
The logical execution binary is unchanged from the rank-placement milestone;
the generated physical metadata identifies `batched_heads_bf16_v1`.

Same checkpoint, devices, TP8/EP8, batch one, context 512 and short prompts as
the preceding milestone. Six runs contain 60 measured intervals, median 29.3441
ms, empirical p95 30.1129 ms, maximum 35.1393 ms. No outliers were removed.
Loading/prefill are excluded. Raw records and hashes are included here; this is
not a concurrency or production latency qualification.
