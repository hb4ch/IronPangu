# Expanded physical decoder and strict storage contracts — 2026-10-16

The compiler expands 1,701 logical operations into **2,661 physical instructions**.
The per-token vendor API graph separates routing projection/top-k, route and
activation quantization, expert output zeroing/GMM/reduction, mHC projection and
Sinkhorn, cache packing, and shared attention metadata. Dense quantization and
query rotary/casts/packing remain explicit. Native execution rejects implicit
storage conversions and undeclared device copies. Preparation-only copies have
declared scopes that are forbidden during capture and after the first token.

Grouped expert scales now load and pack directly as BF16, eliminating their
BF16→FP32→BF16 round-trip. Each rank reserves **12,433,409,088 bytes** in **3,616
fixed arena slots**, including prepared parameters and persistent state; original
checkpoint storage is separate. An unbound bundle requires explicit hardware
preparation. Normal execution rejects missing or incomplete arena manifests.

Validation:

- **69 local compiler/DSL tests** passed, including expansion dependencies and
  shared routing for separate expert projections. Rust was cross-compiled locally.
- Docker native validation passed all eight rank bindings, **23 negative kernel
  contracts**, and **10 negative arena-file contracts**. The NPU prepared-execution
  probe also checked fixed offsets, exhausted capacity, undeclared slot bindings,
  forbidden copies, leases and stream ordering.
- **99 complete output records** matched the unchanged qualified fixture:
  six graph runs (66 records) and three ordinary-execution runs (33 records).
  All **72 rank audits** passed residency checks and independently confirmed one
  arena reservation per rank and **zero device allocations in every token step**.
- Unbound and incomplete bundles were rejected before device initialization.
- Browser checks passed hierarchical pagination, selected kernels, structured
  value fields, casts, preparation details, slot offsets and EP ownership for
  rank 7, without JavaScript errors.

Graph TPOT: **17.7869 ms/token**, median 17.6780 ms, p95 18.6771 ms, or
**56.22 tokens/s**. The prior pre-arena result was 19.9537 ms/token; the matched
OmniInfer reference was 18.1817 ms/token. These are short-context, batch-one
measurements, not production SLOs. Ordinary execution was also correct but slower
at 157.7574 ms/token; graph replay remains the performance path.

The [viewer](execution.html) and `device-program.json` come from the compiler.
The binder adds measured `arena.json` metadata deterministically. `summary.json`,
`runs.json`, and the eager equivalents contain binary, execution and unchanged
fixture hashes. `arena-checks.json` records the additional frozen-allocation audit.

The deployed bundle is `composite-qualified-plan` under
`/data/p00603624/inferfabric/openpangu92` in Docker `inferfabric-92b` on
`7.156.99.58`; the preserved native binary is
`.deploy/openpangu_decode_composite_qualified` in that directory.

The boundary is the vendor API: vendor-internal kernels remain opaque, and
parameter/state preparation remains a labeled recipe rather than a fully lowered
initialization DAG. Arena packing preserves the qualified reuse policy; it is
not yet cost-optimized. INT8/INT4 cache **specifications** are accepted by the DSL,
but compilation rejects them until matching cache/attention kernels are qualified.
No activation host fallback or global fence was introduced.
