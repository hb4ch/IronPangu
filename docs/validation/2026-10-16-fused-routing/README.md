# Fused expert routing — 2026-10-16

**22.1344 ms/token**, 45.1785 tokens/s, versus 23.2259 ms/token previously
(4.7% lower latency). Optimized OmniInfer remains 18.1171 ms/token: native is
**22.2% slower**. Median is 22.1059 ms and empirical p95 22.3471 ms, over 60
intervals in six complete runs.

MoeInitRoutingV3 now filters and sorts the selected routes within each rank's
expert interval, expands and quantizes the input, and emits 32 device expert
counts plus inverse row indices. Count-list grouped matmuls consume these counts.
The packed expert value carries its routing metadata through SwiGLU/down; the
final FP32 reduction uses original scores and inverse indices, skipping -1
foreign routes. No route count or activation is read by the host. Zeroed matmul
output storage defines unused rows, including ranks with no selected expert.

This removes the separate owner lookup, priority multiply, TopK, repeated input
copies, group-table gathers and score-reordering chain. Both paired and separate
expert projection forms remain supported. The logical graph and initialization
plans are unchanged; the [physical viewer](execution.html) records the new
routing, count-list GMM and inverse-index reduction numerical contracts.
The refreshed physical artifact has byte-identical execution.bin to the timed
plan; the refresh adds the final compiler metadata after local validation.

All 33 greedy output tokens match the independent OmniInfer reference. Maximum
selected log-probability differences are 0.011287, 0.030443 and 0.177721. Vendor
quantization/grouping changes logits; previous fixtures remain intact. All 66
graph output records exactly match this qualified eager fixture. All 48 rank
residency, allocation, completion and replay audits pass, with no global fence.

43 compiler tests pass locally, including physical contract inspection and host
fallback exclusion. Native old/new plan validation passes all eight bindings and
nine negative contracts. Rust was cross-built locally; native builds and complete
NPU execution ran inside Docker.

Same checkpoint, devices 0–7, TP8/EP8, batch one, context 512, greedy decoding and
short prompts. Loading/prefill are excluded. Native requests use fresh processes;
OmniInfer uses a warmed engine. Long-context, concurrent-serving and broad quality
qualification remain separate work; this milestone does not establish parity.
