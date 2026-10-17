# Sixteen-token prefill with batched C8 and window attention

2026-10-17, TP8/EP8 OpenPangu 92B, two residents, experimental context 4,096.
This qualifies the narrower short fixture and records a **failed** full long
fixture comparison. It does not promote the variant or qualify 64K serving.

## Correctness and resource checks

Discovery, planned-workspace preparation and frozen execution each match all
123 expected tokens across twelve interleaved requests (arithmetic, geography,
Chinese and a 529-token retrieval, repeated three times). Pages are independently
shuffled. The previously failing 529-token post-EOS continuation now matches the
unchanged budget-16 reference. Standard short requests still use serial prefill
because their prompts contain fewer than sixteen tokens.

The isolated long experiment compares the unchanged independent budget-16 C8
reference at 2,049, 3,072 and 4,088 prompt tokens, with eight forced outputs and
three repetitions. All 72 native tokens are identical to the preceding
sixteen-token implementation. The 2,049/4,088 cases match completely. At 3,072,
the last forced token is 1114 versus reference 148902 in all repetitions; all
outputs through the first EOS match. The qualifier exits nonzero and the
[comparison](comparison.json) retains the failure. Broader short-boundary
numerical differences remain separate unresolved evidence.

All eight frozen native and HTTP rank audits show planned workspace, no growth,
no activation host transfers, no rejected transfers and no retained per-step
allocations/internal fences. Workspace is **131,020,288 bytes/rank**, planned
before arena binding. Arena is **12,611,250,176 bytes/rank**. Compared with the
previous sixteen-token path, workspace grows 103,936 bytes while reduced
metadata storage limits arena growth to 43,008 bytes. The live process HBM sample
is **32,212 MB/rank**; this is a point sample, not a peak-memory measurement.

## Isolated performance

Three repetitions; engine TTFT includes prefill and sampling, excludes loading,
tokenization and HTTP. TPOT uses each generated step's duration directly.
No reference engine or HTTP run overlaps these measurements.

| Prompt tokens | TTFT, ms | TPOT, ms | TTFT reduction vs previous sixteen-token path |
|---:|---:|---:|---:|
| 2,049 | 7,958.508 | 20.053 | 37.86% |
| 3,072 | 11,762.103 | 20.069 | 38.72% |
| 4,088 | 15,742.541 | 20.064 | 38.71% |

The current sixteen-token variant reduces TTFT by about 48% relative to the
current four-token variant. These are native-version comparisons, not a claim
of OmniInfer performance parity. See [performance.json](performance.json).

## Concurrent HTTP

vLLM Rust 0.25.1, `max_num_seqs=2`, token budget 32, packed prefill and serial
decode. Full standard text/SSE outputs, six concurrent requests, queue isolation,
disconnect recovery, cancellation during a 4,088-token peer prefill, overflow and
unsupported-sampling rejection, and graceful SIGTERM shutdown all pass.

The long request's HTTP TTFT is **16.872 s**. Its short peer finishes first with
**1.079 s TTFT** and streamed text intervals of **82.0–82.9 ms**, improved from
1.508 s/about 121 ms, but still substantially delayed by fixed-size chunk work.
Text chunks are not necessarily individual model tokens. There are 29 mixed
scheduler rounds, 256 prefill graph replays/rank, 17 admissions and zero final
occupied page bytes. These tests do not provide a production latency SLO.

## Reproduction and provenance

Compile `validation/2026-10-17-long-index-selection/model.inferfabric` with
`--resident-sequences 2 --prefill-tokens 16 --shared-pages`. Discover every
workspace request, plan the maximum, prepare again and bind the frozen arena.
Use `validation/2026-10-17-prefill16/reference-budget16.json`: first four cases
for preparation/interleaving, last three for isolated long testing. Never replace
expected outputs with native results. All runs opt into experimental long context.

- Native library SHA256: `cf8151d17fd727528f2823ca1705c6a38d175da6bf7bb7bd5be49f9873ed11c2`.
- Execution binary SHA256: `fa56a54d2c5623044a52e97592e9a098ce8af5aaad5d1d29e27f4ebb2622fa53`.
- Rust frontend SHA256: `a81c91652b82e4917b8db7e767138c51f75fc3f2baa10ce799d476ae37ec582d`.
- Remote raw artifacts: `window16-{source,discovery,workspace,preparation,bound,qualified,long,http}`
  under `/data/p00603624/inferfabric/openpangu92` in `inferfabric-92b` on `7.156.99.58`.
- Committed compact audits retain source hashes; raw local copies are under
  `.deploy/window16-evidence`.

Next: the diverse 8K/16K/32K/64K reference ladder and a prepared family of chunk
sizes for mixed-workload latency. The serving admission gate remains unchanged.
