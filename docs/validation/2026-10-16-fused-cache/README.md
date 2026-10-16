# Fused cache preparation — 2026-10-16

**23.2259 ms/token**, 43.0554 tokens/s, versus 23.9198 ms/token previously
(2.9% lower latency). Optimized OmniInfer remains 18.1171 ms/token: native is
**28.2% slower**. Median is 23.1907 ms and empirical p95 23.5721 ms, over 60
intervals in six complete graph runs.

The DSL explicitly selects `cache_norm_rope_append`: RMS normalization,
half-rotary and paged cache write become one vendor invocation. The compiler
validates 512 latent values, 64 unrotated key values, 128 sinks, epsilon, theta,
and the gamma binding. The cache handle retains its single-consumer restriction.
The expanded plan has 1,699 nodes, one gather and 92 explicit all-reduces.
All eight initialization binaries are unchanged. The original separate operators
remain supported, with old/new native plan binding tests passing.

The [generated viewer](execution.html) identifies the fused kernel and numerical
contract. Gamma, input and rotary coefficients are BF16; coefficients and INT64
position are uploaded as controls once per token. Activations and cache contents
never cross the host. The vendor API requires output descriptors even for this
cache-only consumer; the unused outputs use the reusable device pool. No global
fences or host fallback are introduced.

All 33 greedy tokens match the independent OmniInfer reference. Maximum selected
log-probability differences are 0.030360, 0.004399 and 0.177752. Logits differ from
the separate FP32-rotary path; previous fixtures remain intact. All 66 graph
records exactly match this independently qualified eager fixture, and all 48
rank residency, allocation, completion and replay audits pass.

Local validation: 43 compiler and 16 DSL tests, including wrong input shape,
normalization epsilon and rotary theta rejection. Native validation: eight rank
bindings and nine negative contracts on each old/new plan. Rust compilation was
local; native builds and full NPU generation ran inside Docker.

Same checkpoint, devices 0–7, TP8/EP8, batch one, context 512, greedy decoding and
short prompts as prior milestones. Loading/prefill are excluded. Native requests
use fresh processes; OmniInfer uses a warmed engine. This is not a long-context,
concurrent-serving or broad quality qualification. Performance parity remains open.
