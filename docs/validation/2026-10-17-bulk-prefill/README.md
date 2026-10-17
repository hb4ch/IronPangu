# Batched prefill: full 92B qualification — 2026-10-17

The uncaptured program now lowers head projections to one batched matmul,
followed by an explicit head-to-token layout operation. Gather/repeat operate
on whole token matrices. Cache normalization/rotary writes, index-cache writes,
and sparse-attention metadata preparation consume vectors of query positions.
They no longer submit one operation per prompt token. Single-token decode
retains its graph path and avoids the extra head-layout materialization.

## Evidence

Run on 7.156.99.58, Docker `inferfabric-92b`, checkpoint `/models/92B`,
TP8/EP8, plan `bulk-prefill4k-source`, binary
`native-build-head-batch/program_engine_probe`. The persistent lifetime
qualification used preparation mode, permuted pages, packed prefill,
experimental long context, output diagnostics and first-EOS comparison.

- `process.json`: native return code 0, 140.93 seconds for the qualification.
- `results.json`: 21 requests, seven prompt cases repeated three times;
  lengths 12, 15, 14, 529, 2049, 3072 and 4088. All 159 tokens through first
  EOS match the unchanged reference. Forced continuation after EOS is excluded.
- `repeatability.json`: all seven cases repeat identically within that scope.
- `transfers/`: all eight ranks report zero activation host transfers.
- Per rank: 21 request forwards consume 29,337 prompt tokens, plus two
  4096-token preparation forwards. This is actual batched execution.

4088-token engine TTFT is 6.829, 6.754 and 6.661 seconds; corresponding
decode TPOT is 24.822, 25.961 and 38.079 ms. These are three serial requests,
not a throughput benchmark. The preceding rowwise implementation measured
107.50–114.81 seconds TTFT on this case.

Before this full run, production-helper NPU probes checked head projections
and layout (18 cases, 29,445,120 values), BF16/C8 fused cache writes
(20 launches, 9,775,872 values), index key/scale writes (six cases,
2,106,368 keys), and metadata (15 launches, 6,517,248 indices). They include
4096-token batches, tails and strided/permuted cache positions. Metadata
coverage caught and fixed concurrent scalar writeback sharing a cache line.
Compiler tests: 22 passed. Native plan validation: 2862 nodes, eight rank
bindings, 36 rejected negative contracts.

## Remaining scope

This run uses preparation mode and serial decode. It does not qualify frozen
arena replay, cross-request batching, multi-chunk continuation, HTTP/APC
integration or 64K. The next priority is actual multi-sequence execution and
continuous batching, followed by those longer-context serving gates.
