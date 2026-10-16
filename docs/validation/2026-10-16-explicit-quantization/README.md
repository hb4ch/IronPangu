# Explicit runtime quantization — 2026-10-16

The physical compiler expands each dense INT8 projection into an explicit
`quantize` instruction followed by `linear`. The former emits signed INT8 data
and FP32 per-token scales using `aclnnDynamicQuant`; the latter consumes that
pair with `aclnnQuantMatmulV5`. The physical executor rejects an unquantized
input rather than quantizing implicitly. The 92B program has 2,027 instructions.

DSL authors can specify floating-point casts, runtime quantization, and cache
precision contracts. INT8/INT4 cache execution is not yet supported by the
qualified attention backend; compilation fails explicitly for those contracts.

Local compiler/DSL tests and cross-compilation passed. Docker native validation
passed eight rank bindings and 18 negative contracts. Six full 92B graph runs
matched all 66 qualified output records, including diagnostic scores; all 48
rank residency/prepared-execution audits passed. The existing fixture was not
changed. TPOT is **19.9537 ms/token**, effectively unchanged from 19.9519 ms.
Hashes and measurements are recorded in `summary.json` and `runs.json`.

This milestone expands dense quantization. Other composites and the frozen
memory arena remain unfinished; it does not claim a complete primitive DAG.
