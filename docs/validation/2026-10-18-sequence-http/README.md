# Frozen continuous batching over HTTP — 2026-10-18

PASS on 7.156.99.58, Docker `inferfabric-92b`, TP8/EP8 OpenPangu 92B,
vLLM Rust 0.25.1. Qualifier process exit 0. This qualifies the 4K context
fixture envelope, not multi-chunk or 64K serving, APC, or sustained load.

The previous HTTP run failed at instruction 472 requesting 15,728,640 bytes.
Its sampled binding had only one sufficiently large eligible slot (25,165,824
bytes). Sampled address membership was too restrictive for dynamic tail reuse.
The binder now emits equal-capacity equivalence classes explicitly in
`arena.json`, its embedded viewer data and each rank's binding file. Original
observations remain in `observed_instruction_slots`. The permanent workspace
is excluded from expansion. Runtime still takes only free, eligible slots on
the owning stream and fails on exhaustion; no capacity, offsets, runtime
binary, allocation fallback or synchronization policy changed.

Evidence: [results](results.json), [server log](server.log),
[eight-rank resource and graph summary](rank-summary.json).
The raw results' legacy `mode` label says serial decode; the native counters
show the actual sequence decode graphs: 103 single-sequence and 35 two-sequence
replays per rank (including startup captures). There are no prefill graphs.

- Four baseline fixtures pass collected/SSE text and token-usage comparison.
- Eight queued concurrent requests pass output isolation.
- 2,049/3,072/4,088-token prompts pass reference output through first EOS while
  sharing the server with cancellation and a short request. Forced output
  after EOS is outside the requested comparison scope.
- 27 admissions; 13,833 completed input tokens fully accounted by arithmetic
  histories; two active residents; ten mixed prefill/decode ticks.
- Shared pages fully reclaimed at shutdown; unsupported sampling and context
  overflow rejected; disconnect recovery and graceful shutdown pass.
- Every rank: frozen arena 38,116,810,768 bytes (unchanged), fixed workspace
  159,221,248 bytes, zero workspace growth, activation host transfers,
  rejected transfers, recorded step allocations or internal fences.

HTTP TTFT for the mixed 2,049/3,072/4,088 cases was 8.987/6.846/6.530 seconds.
These are individual observations including preparation/cache effects and
concurrent scheduling, not a throughput benchmark. Baseline stream text-chunk
intervals are mostly 22–30 ms. Mixed-request gaps include competing prefill
(up to 2.59 seconds), so they must not be reported as kernel decode TPOT.
SSE text chunks are also not necessarily one token. Longer-context TTFT/TPOT
qualification remains a separate gate.

Reproduction, in the existing Docker work directory with CANN/vendor environment:

```sh
python3 scripts/bind-prepared-arena.py sequence-serving-fixed-plan \
  sequence-serving-fixed/prepared sequence-capacity-frozen-plan
python3 scripts/qualify-program-http.py \
  --server /data/p00603624/inferfabric/inferfabric-server-sequence \
  --plan sequence-capacity-frozen-plan --checkpoint /models/92B \
  --library native-build-head-batch/libinferfabric_program.so \
  --cases batched16-preparation-reference.json \
  --resident-cases batched16-long-reference.json \
  --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --expect-prefill packed --respect-eos --check-arithmetic-history \
  --output sequence-capacity-http --port 18085
```

Set `INFERFABRIC_DECODE_GRAPH=1`, `INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1`,
and the existing HCCL environment. Output paths must be new. Local Python
unit tests: nine pass. The C++ bounded-buffer regression ran in Docker and
passed all 4,096 tail sizes plus live-slot/capacity-class isolation checks.
