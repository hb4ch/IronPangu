# Borrowed operands and MLA packing — 2026-10-16

**21.4350 ms/token**, 46.6526 tokens/s, versus 22.1344 ms/token previously
(3.2% lower latency). Optimized OmniInfer remains 18.1171 ms/token: native is
**18.3% slower**. Median is 21.4141 ms and empirical p95 21.7062 ms, over 60
intervals in six complete runs.

mHC pre/post, attention, router projection and SwiGLU borrow read-only device
operands instead of copying into staging buffers. mHC retains its residual's
shared allocation lease through post. Recycling and execution remain ordered on
the owning stream. MoME still stages its input because its vendor kernel mutates
that input; the change does not introduce unchecked in-place aliasing.

MLA rotary extraction uses one bounded strided cast instead of six copies.
One column concatenation packs the absorbed query and rotated channels instead
of twelve head-by-head copies. FP32 rotary and BF16 rounding are unchanged.
The [generated viewer](execution.html) exposes these implementation decisions.
Logical execution and initialization plans are unchanged.

The complete three-prompt eager fixture is byte-identical to the previous
milestone (SHA256 `c1c7740653cd408dc8a1377b2adc0a251b978f24c6c1c4d00cb9b7f571dfbc35`).
All 33 greedy tokens match the independent reference. All 66 complete graph
records match eager, and all 48 rank residency, allocation, completion and replay
audits pass. No activation readback, implicit host fallback or global fence.

43 compiler tests pass locally. Native binding validation passes on the 1,699-node
current and 1,837-node prior plan, including all eight ranks and nine negative
contracts each. Rust was cross-built locally; native builds and full NPU tests
ran inside Docker. The NPU process inventory showed only this test workload on
devices 0–7.

Same checkpoint, TP8/EP8, batch one, context 512, greedy decoding and short prompts.
Loading/prefill are excluded. Native requests use fresh processes; OmniInfer uses
a warmed engine. This is an offline decode result, not broad quality,
long-context or concurrent-serving qualification. Performance parity remains open.
