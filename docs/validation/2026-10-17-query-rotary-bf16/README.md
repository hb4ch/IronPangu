# Explicit BF16 MLA query rotary

Date: 2026-10-17. Status: precision contract implemented and standard full-model
regression passed; reference parity remains unresolved. Context stays at 512.

The reference's `_q_rope` applies rotary to BF16 query columns. Native MLA query
lowering previously always materialized those columns in FP32, rotated them and
cast the result to BF16. The new optional `rotary_dtype` makes this numerical
choice explicit. The experimental 92B learned-index DSL selects BF16; existing
DSLs retain their FP32 default. BF16 currently requires rank-local query input.

The BF16 expansion uses existing generic primitives: `query_project`,
`column_slice(rows=6,start=128,width=64)`, `rope_bf16`, `concat_columns(rows=6)`.
The compiler/viewer show BF16 intermediate storage and coefficients, without a
hidden promotion or rounding cast. No native executor/kernel changes were needed.
This removes 46 physical instructions: 2,814 becomes 2,768 for the 46-layer plan.

Local compiler token tests passed (11 tests); targeted regression also checks
all 46 BF16 query expansions, the preserved FP32 default and invalid dtype
rejection. Native plan validation checked all eight rank bindings and rejected
33 invalid contracts. Rust was locally cross-compiled; native execution was
inside `inferfabric-92b` on `7.156.99.58`.

Full TP8/EP8 output results:

- Standard fixtures: **99/99 tokens passed** during discovery, planned-workspace
  preparation and frozen-arena execution, using packed prefill, interleaved
  resident requests and shuffled pages.
- All **336 boundary tokens** were produced and repeat exactly across three
  repetitions. Every sequence equals the previous FP32-query-rotary native run.
- Exact reference comparison still fails: **9/14** sequences match the original
  reference and **10/14** match the one-token-prefill reference. This change does
  not fix the remaining divergence, and no expected tokens were changed.
- All eight rank audits pass: zero activation host transfers, workspace growth,
  replay allocations or internal completion fences; frozen arena and ACLGraph
  remain active.

Arena: **12,451,957,248 bytes/rank**, 6,656 bytes smaller than the prior arena.
Shared workspace stays **130,916,352 bytes/rank**. Interleaved timings in
`native-results.json` include peer work and are not isolated TPOT measurements.
No new HTTP or 64K performance claim is made.

Evidence: `native-results.json`, `native-audit.json`, `boundaries-audit.json`,
`boundary-comparison.json`. The reference captures remain linked from the
[parity investigation](../2026-10-17-output-parity/README.md).

SHA-256:

- Execution: `b594cde84eb73e477ce4e5789753103f4e70d0e33791b6094fbbd968a61e7cd8`
- Native library: `65188f8c091b827bc4e65ee13cfccc244765ad43f48bd87c6f5bf81be82f2175`
- Reference `npu_pangu.py`, equal in local source and remote Docker:
  `66f933516fa80ef126ad9ce708f34ae268d569fa5827cfd3a418006ad88ae256`

The next controlled comparison is collective precision. Current native DSL
defaults to FP32 AllReduce; reference paths reduce BF16 outputs. This is already
expressible through the generic `allreduce(..., dtype="bf16")` contract. Test it
separately, retaining both reference captures and the current failure evidence.
