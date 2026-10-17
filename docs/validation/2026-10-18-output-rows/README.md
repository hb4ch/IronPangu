# Explicit sampled-row output execution — 2026-10-18

IFTOKN08 output-row execution passes full 92B discovery, fixed workspace and
frozen arena qualification: each stage matches 39/39 reference tokens over six
interleaved 8K/short requests. Pure decode graphs replay. Frozen audits on all eight ranks show
zero activation-host transfers, execution-time allocations, workspace growth or
internal completion boundaries. Transient descriptors/invocations drain to zero.

The logical model graph is identical to `owned-opapi-source/execution.json`.
The declared suffix is instructions 2858–2861: norm, BF16 vocabulary projection,
rank-major AllGather and token-layout conversion. Source 2857 has width 2560.
Uncaptured batches select only requested output rows on device. Zero-row calls
still execute the entire state-updating prefix. Decode graphs retain their fixed
row geometry. Compiler and reader checks protect the row/layout boundary.

The frozen arena is **38,009,938,432 bytes/rank**, versus 40,492,359,704 in the
previous metadata-lifetime backend: **2,482,421,272 fewer bytes/rank**. Shared
workspace stays 159,221,248 bytes/rank. This is arena reservation, not a complete
HBM occupancy measurement. Timing samples in the trace are not a controlled
performance comparison. 64K HTTP/APC and a new collective capture remain pending.

Local checks: 14 compiler unit tests, 23 token bundle/format tests and three
physical-program tests pass; the ARM Rust cross-build passes. Native builds run
inside Docker. The generated viewer passes the existing kernel/collective
inspection regression and carries the explicit selection/row-domain metadata.

## Reproduction

Use `INFERFABRIC_EXPERIMENTAL_OUTPUT_ROWS=1` with the locally cross-compiled
`inferfabric-compiler-output-rows` to compile `output-rows.inferfabric` (the
canonical learned-C8 92B DSL with context 65536), three residents, 4096 prefill
tokens and shared pages. The experiment remains opt-in at compilation.

Remote workspace `/data/p00603624/inferfabric/openpangu92`, Docker
`inferfabric-92b`, normal CANN and vendor environment, HOST expansion, HCCL base
port 42400. Build/source: `native-build-output-rows` / `native-output-rows-source`.
Run `qualify-sequence-serving.py --batch-trace` with the new program probe,
`output-rows-source-v2`, `/models/92B`, existing `device-zero-apc8k-fixture`
requests/bindings/reference and output prefix `output-rows8k`.

The resulting bundle is `output-rows8k-frozen-plan`. `summary.json` records
library/execution hashes and metadata audits. The frozen directory preserves raw
tokens, completed batch geometry, process result and all per-rank resource audits;
discovery/fixed summaries preserve earlier gate results. No earlier arena was
retagged for the new binary.
