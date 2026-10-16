# Paired BF16 projections — 2026-10-16

**20.0132 ms/token**, 49.9671 tokens/s, versus 21.1723 ms/token previously
(5.5% lower latency). Median is 19.9138 ms and empirical p95 20.9434 ms over
60 intervals in six complete graph runs. The refreshed OmniInfer reference is
18.1817 ms/token: native remains **10.1% slower**.

Generic `linear` now accepts `paired_weight` and `paired_width` for two
replicated BF16 projections sharing an input. Their rows are packed once on
device, and one matmul returns the concatenated outputs. Both binding roles
and the additional parameter bytes are exposed in the generated viewer.
Non-replicated pairs, unsupported dtypes and malformed widths fail validation.

The DSL pairs query/KV compression in 46 layers and gate/up projections in 44
shared experts, eliminating 90 dense matmul launches per token. One-input
`swiglu` consumes equal packed halves directly, eliminating 44 activation
concatenations. The two-input form remains supported. The new expanded graph
has 1,701 nodes, one gather and 92 explicit FP32 all-reduces.

Packing costs **799.375 MiB per rank** in this model, in addition to the retained
source parameters. It occurs before capture and never transfers activations to
the host. Eliminating that source/packed duplication is future initialization
planning work. All eight initialization binaries are byte-identical to the
prior plan; only the execution graph changes.

The complete eager fixture is byte-identical to dense attention, SHA256
`7a9eafc79473c5843741a3c657d692374d71f8b658f65e00f3ae5fa28a8a6cab`.
All 33 greedy tokens match the independent reference. All 66 complete graph
records match eager, and all 48 rank residency/allocation/completion/replay
audits pass. No activation readback, implicit host fallback or global fence.

61 compiler/DSL tests pass locally. Native validation passes all eight bindings
and 13 negative contracts on the new plan, plus ten negative contracts on the
prior plan. Rust was locally cross-built; native builds and NPU execution ran
inside Docker. The [viewer](execution.html) includes final memory descriptions;
its execution binary is byte-identical to the measured plan.

Same checkpoint, devices, TP8/EP8, batch one, context 512, greedy and short
prompts. Native uses HOST collectives; the AIV experiments were slower and did
not change the default. Loading/prefill are excluded. These offline measurements
do not establish long-context quality, concurrent serving or a production SLO.
