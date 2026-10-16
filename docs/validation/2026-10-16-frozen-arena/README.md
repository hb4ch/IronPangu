# Hardware-bound arena and query expansion — 2026-10-16

MLA query construction now expands into five executable instructions: BF16
projection, strided BF16-to-FP32 extraction, FP32 rotary embedding, BF16 rounding,
and column concatenation. The full model has 2,211 physical instructions.

The prepared arena reserves **12,459,363,904 bytes per rank**, with **3,705**
fixed slots. This includes packed parameters, persistent state, activation
temporaries and shared vendor workspace; original checkpoint allocations are
separate. Capacity is deliberately conservative and preserves the qualified
reuse policy. The native executor checks program/backend identity and rejects
growth or instruction-to-slot bindings absent from hardware preparation.

All **66 complete output records** matched the unchanged qualified fixture
across six full 92B graph runs. All **48 rank audits** passed residency, warmed
allocation and completion checks. Native contract validation passed eight rank
bindings and 18 negative cases; the arena parser passed ten negative cases.
Local compiler/DSL tests and Rust cross-compilation passed. Browser checks
verified kernel details, arena offsets and rank selection without script errors.

Pooled TPOT is **17.6477 ms/token** (56.66 tokens/s), versus 19.9537 ms before
query expansion and arena binding. The same-setup OmniInfer reference is
18.1817 ms/token. This comparison covers the existing short-context, batch-one
fixture; it does not establish production throughput or latency guarantees.

`summary.json` records the binary/plan/fixture hashes. `arena.json` records the
hardware preparation bindings; `execution.html` is the deterministic compiler
viewer with that bound metadata. No activation host fallback or global fence was
added. Other composite recipes still need expansion; this milestone does not
claim completion of the entire primitive execution DAG.
