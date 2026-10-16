# Dense sink attention — 2026-10-16

**21.1723 ms/token**, 47.2314 tokens/s, versus 21.4350 ms/token previously
(1.2% lower latency). Median is 21.0707 ms and empirical p95 21.9877 ms over
60 intervals in six complete graph runs. OmniInfer's earlier optimized result
is 18.1171 ms/token; this native result remains **16.9% slower**.

The validated `all_causal_within_window` attention contract permits a dense
sink-attention kernel: context is bounded by both 512 and the declared window.
The backend uses FusedInferAttentionSink's TND sliding-window mode, with
`pre_tokens=context-1`, `next_tokens=0`, and separate learned sink tensors.
This does not implement indexed DSA or extend the context limit.

The combined paged latent/rotary cache is accessed through bounded strided
views. Immutable sink views are packed once on device (144 KiB per distinct
cache, 6.46875 MiB per rank for this model). A shared 4 MiB causal mask is an
initialization constant. One AICPU operation updates the 4 KiB device metadata
buffer per rank/token; graph replay includes that operation. Sequence length
is a control input. Candidate-index uploads are eliminated. There is no
activation readback, implicit host fallback, or device-global fence.

All 33 greedy tokens in the full three-prompt eager run match the independent
reference. The kernel changes floating-point reduction order: maximum selected
log-probability deltas against reference are 0.034083 (arithmetic), 0.030453
(geography), and 0.225326 (Chinese). This is not bitwise logit equivalence or
broad quality qualification. The eager fixture SHA256 is
`7a9eafc79473c5843741a3c657d692374d71f8b658f65e00f3ae5fa28a8a6cab`.

All 66 complete graph records match the eager fixture. All 48 rank residency,
allocation, completion and replay audits pass. The 1,699-node logical binary is
unchanged. The [generated viewer](execution.html) exposes the physical kernel,
metadata generation, persistent buffers and permitted transfers. Local compiler
and DSL tests pass; Rust was cross-built locally. Native builds and full model
runs used the remote Docker container.

Same checkpoint, eight devices, TP8/EP8, batch one, context 512, greedy decoding
and short prompts. Native still uses `HCCL_OP_EXPANSION_MODE=HOST` for this
milestone. Inspection found that OmniInfer defaults its communicators to AIV
when no expansion mode is set; matching that setting is the next experiment.
Native requests use fresh processes, OmniInfer a warmed engine. Loading/prefill
are excluded. This is an offline decode benchmark, not a serving SLO.
