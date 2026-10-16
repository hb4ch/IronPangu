# Fused expert-output reduction — 2026-10-16

**28.4010 ms/token**, 35.2101 tokens/s, down from 29.5091 ms/token (3.8%).
The matched OmniInfer reference remains 18.1171 ms/token: a 1.57× gap remains.

MoeFinalizeRoutingV2 replaces separate FP32 multiplication and ReduceSum over
eight owner-ordered expert rows. Its identity row indices are immutable device
control data, initialized once; per-token scores and expert outputs stay on NPU.
FP32 input/accumulation/output and the following FP32 all-reduce are retained.
The intermediate product tensor is eliminated. Physical metadata identifies
`fused_finalize_fp32_v1` and exposes the device index/reduction contract.

All three full-model eager results are byte-identical to the batched-head
fixture (SHA256 `812b4df8f408db68bfb6ff6e618544ea51a22bba4bf7341730b9fa96a2120d7e`).
All 33 greedy tokens match the independent OmniInfer reference; selected-token
log-probability differences are unchanged. Six graph runs pass all 66 complete
output records and 48 rank residency/replay/allocation/completion audits.
No activation readback, implicit fallback or global fence was added.

All 42 compiler tests pass; Rust cross-build is local and native compilation/
execution is inside Docker. Same checkpoint, devices 0–7, TP8/EP8, batch one,
context 512 and short prompts as preceding milestones. Sixty intervals give
median 28.3075 ms and empirical p95 29.1604 ms. Loading/prefill are excluded;
the evidence does not establish concurrent or long-context serving performance.
