# Prepared immutable weight layouts — 2026-10-16

**26.4094 ms/token**, 37.8653 tokens/s, versus 28.4010 ms/token before packing
immutable MLA weights once (7.0% lower latency). The matched optimized OmniInfer
reference remains 18.1171 ms/token, leaving a **1.46× gap**.

`PreparedWeightViews` is a reusable device-only cache of bounded strided immutable
parameter views. It materializes each layout once on the owning prepared stream
and retains the resulting allocation across graph replays. It is not an activation
cache or a checkpoint loader. The permanent initialization executor still owns
checkpoint loading and parameter identity. The caller must retain immutable
source parameters for the cache lifetime.

MLA query/value weight views now use contiguous batched layouts, avoiding repeated
vendor layout materialization. This costs 786,432 bytes per distinct view, or
69 MiB per rank for this model's 92 views, in addition to the source parameters.
Generated physical-plan boundaries expose the preparation and memory cost.
The rank-0 audit records exactly 92 `immutable_layout_copy` preparations across
the complete request. No parameter host transfer or new completion boundary is
introduced. Generic tests validate gapped, transposed layout contents on NPU
and repeated cache lookup; existing bounds/lifetime tests continue to pass.

All three eager result files are byte-identical to the preceding batched-head
and fused-reduction fixture. All 33 greedy tokens match the independent reference.
All six graph runs pass 66 complete output-record comparisons and 48 rank
residency/replay/allocation/completion audits. Forty-two compiler tests pass
locally; Rust cross-build is local, native build and NPU tests run inside Docker.

Same checkpoint, devices 0–7, TP8/EP8, batch one, context 512 and three short
prompts as prior milestones. Sixty intervals yield median 26.3301 ms and empirical
p95 27.1821 ms. No outliers are removed. Loading/prefill are excluded; this is
an offline decode comparison, not a concurrent-serving or long-context SLO.
