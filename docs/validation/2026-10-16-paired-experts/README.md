# Paired expert projections — 2026-10-16

**23.9198 ms/token**, 41.8064 tokens/s, versus 25.7072 ms/token previously
(7.0% lower latency). OmniInfer remains faster at 18.1171 ms/token: a **1.32×
latency gap** remains. Median is 23.8984 ms and empirical p95 24.1438 ms over
60 intervals across six complete runs.

The DSL explicitly binds gate and up weights in one `expert_linear` operation.
The backend prepares a paired bank directly from those bindings, quantizes the
shared input once, and launches one sparse grouped matmul. Packed output flows
directly into fused SwiGLU/down without intermediate row repacking. This removes
44 graph nodes and 44 grouped matmuls per token. The plan now has 1,837 nodes,
one gather and 92 explicit all-reduces. The original separate-projection form
remains accepted. Initialization binaries are byte-identical for all eight ranks.

The [generated viewer](execution.html) exposes both weight bindings and the
selected paired kernel's precision/layout contract. No checkpoint names select
this optimization implicitly, and no activation moves to the host.

All three full-model eager sequences (33 tokens) match the independent OmniInfer
reference. The complete eager fixture is byte-identical to the previous milestone,
including log probabilities (SHA256
`c8cce5322e0f41f96214dd2cc0e452495dd11f6b4480456e7be2c4aa8b85f41d`).
All 66 graph records match eager exactly; all 48 rank residency, allocation,
completion and replay audits pass. No global fences are added.

Local validation passes 43 compiler and 16 DSL tests, including paired binding
validation and rejection of mismatched routing provenance. Native validation
passes all eight bindings and nine negative contracts for both old and new plans.
Rust is cross-built locally; native builds and NPU execution run inside Docker.

Same checkpoint, devices 0–7, TP8/EP8, batch one, context 512, greedy decoding and
short prompts as prior milestones. TPOT excludes loading/prefill. Native requests
use fresh processes; OmniInfer uses a warmed engine. This is an offline decode
comparison, not a concurrent-serving or long-context production qualification.

## Follow-up profiling

The fresh [device trace summary](device-profile.json) records 88 grouped matmuls
per token, matching the reference launch count. mHC preprocessing and attention
remain the largest compute entries. Inclusive sums are not additive TPOT and the
reference trace used another prompt.

The reference trace identifies mHC Pre V2. Switching native to that API with
batchInvariant=false and inference-only outputs passed full output and replay
audits, but measured [23.9141 ms/token](mhc-v2-experiment-summary.json) versus
23.9198: no meaningful gain. The V1 implementation is retained. This experiment
does not establish why the profiled mHC durations differ.
