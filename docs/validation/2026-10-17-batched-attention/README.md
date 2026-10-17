# Batched packed-C8 attention

Date: 2026-10-17. Packed-C8 attention now executes one TND call for an entire
prefill chunk. Both learned-index and bounded causal-index paths consume their
per-row indices directly and return one contiguous BF16 output. There is no
per-query output assembly, activation host transfer or added fence.

The physical recipe/viewer exposes the call count, output shape, zero assembly
copies, causal alignment and initialization-only query-count control array.
Queries are `[T,6,576]`, outputs `[T,6,512]`. The operator consumes total query count
`T`, the final row's device KV length and each query's own candidate list. Metadata
generation and cache appends remain ordered on the owning stream before attention.

## Independent operator validation

The extended `packed_cache_probe` validates query counts 1, 4 and 16 against a CPU
oracle using the actual packed latent scales, BF16 rotary values and learned-sink
layout. Cases cover initial causal lengths, positions 31/32 across a page boundary,
and position 511. Distinct query rows are checked independently. Each shape also
passes eight exact eager-versus-ACLGraph replays. Diagnostic output reads belong
only to the probe, not serving execution. [Evidence](operator-probe.json).

| Query rows | Values checked | Maximum absolute error | Attention workspace |
|---:|---:|---:|---:|
| 1 | 36,864 | 0.00120384 | 93,970,944 bytes |
| 4 | 147,456 | 0.00130768 | 99,869,184 bytes |
| 16 | 589,824 | 0.00146597 | 108,913,152 bytes |

The tolerance is `0.003 + 0.02 * abs(expected)`. The probe's query count 2 is
accepted by its CLI but was not part of this recorded hardware run. These are
512-cache-row operator cases, not full-model 64K qualification.

## Full 92B results

All 99 standard tokens pass discovery, preparation and frozen execution with
two interleaved residents and permuted page bindings. The short frozen audit
reports 106 decode and 28 prefill graph replays. Packed-C8 attention preparations
fall from 240 to 96; for each four-token model chunk, sixteen attention layers
now issue 16 calls instead of 64 and remove 64 assembly copies.
[Short results](short-results.json), [audits](short-audit.json).

All 72 long-run native outputs are identical to the previous four-token
implementation, across three repetitions of 2,049-, 3,072- and 4,088-token prompts.
All match the independent budget-four reference through EOS. The known forced
post-EOS difference at 4,088 remains and the full-reference qualification still
returns failure. This optimization does not resolve numerical parity or raise
context admission. [Comparison](comparison.json).

| Prompt tokens | Previous mean TTFT | Batched attention TTFT | Reduction | Mean TPOT |
|---:|---:|---:|---:|---:|
| 2,049 | 19,064.159 ms | 17,490.985 ms | 8.25% | 20.064 ms |
| 3,072 | 28,549.600 ms | 26,059.916 ms | 8.72% | 20.124 ms |
| 4,088 | 38,050.014 ms | 34,691.198 ms | 8.83% | 20.074 ms |

Warm loaded model, serial requests, four-token prefill, eight forced outputs and
three repetitions. TTFT excludes model loading, HTTP and queueing. No reference
or HTTP workload ran concurrently with the measured native test. These separate
runs establish measured improvement, not statistical confidence intervals.
[Before/after](performance.json). Do not compare these as sixteen-token results:
only the operator probe, not this full-model experiment, exercised sixteen rows.

Arena size is 12,536,875,008 bytes/rank, 512 bytes above the prior implementation.
Maximum shared workspace remains 130,916,352 bytes/rank. All eight
[long-run audits](long-audit.json) report zero activation host transfers, workspace
growth, retained-step allocations and internal completion boundaries. Live sampled
process memory remains 32,120 MB/rank; this is not a peak measurement.

## Rust HTTP regression

The vLLM Rust 0.25.1 frontend passes full fixture text, SSE agreement, six
concurrent requests, queue isolation and disconnect recovery. Cancellation during
a 504-token peer prefill preserves both outputs; the short request finishes first.
The scheduler exercises two active residents, token budget eight and 25 mixed
ticks. All 17 admissions retire their pages, with zero final occupancy; context
overflow and unsupported sampling are rejected. SIGTERM drains and closes workers.
[Results](http-results.json), [audits](http-audit.json). All eight HTTP audits also
pass device residency, frozen arena, no scratch growth and no internal fences.

The mixed short request's text chunks arrive about 53.4–53.9 ms apart; these are
HTTP text-chunk intervals, not guaranteed per-token intervals. This regression
uses the experimental 4K bundle but executes prompts only through 504 tokens.
It does not establish concurrent 4K behavior for this changed backend, batched
multi-request decode or prefix reuse. HTTP artifacts are `batched-attention-http`.

## Reproduction details

Build the Rust compiler/frontend locally. Build and run native targets in Docker
`inferfabric-92b` on 7.156.99.58, working directory
`/data/p00603624/inferfabric/openpangu92`. The operator commands are
`native-build/packed_cache_probe 1`, `4` and `16`. Compile the prior
[4K learned-index DSL](../2026-10-17-long-index-selection/model.inferfabric) with
two residents, shared pages and four prefill tokens. Run discovery, maximum shared
workspace planning, preparation and frozen arena binding again; the backend
fingerprint rejects older bindings. Artifacts use prefix `batched-attention-`
(`source`, `discovery`, `workspace`, `preparation`, `bound`, `qualified`,
`long-qualified`). Long qualification uses the unchanged
`index-select-reference-c8-chunk4.json` with `--permuted-pages --packed-prefill
--experimental-long-context --output-diagnostics`. Retain its nonzero result.

Sixteen local compiler tests pass, including physical invocation counts,
materialization and shape checks for both C8 selection modes at chunk sizes
2/4/16. The query-count initialization transfer is explicitly documented in the
physical recipe. The final description refinement does not change execution.bin.

| Artifact | SHA256 |
|---|---|
| Native library | 60080405369661b36367476108cc564245c050a9a60b2cb6300975e5bf050883 |
| execution.bin | 3b0c4789a6743ffdbc5b22e9d4b7dd67651e97b0840f0559945ba178169d1084 |
| Rust frontend | 22d91395d5a65c2ce83c0037f6e1ebdcd8ea1465ebfe2a15788d20dabcd2fa08 |

Next: batch BF16 sliding-window attention, qualify larger full-model chunks and
compile smaller/larger prefill variants for measured mixed-workload latency
budgets. General accuracy, multi-request batched decode, hybrid prefix restore
and robust 64K serving remain incomplete. Default context admission remains 512.
