# Owned API backend: 64K serving qualification — 2026-10-18

Both full-model runners exited 0 using `native-build-owned-opapi` and the
already qualified `owned-opapi-raii8k-frozen-plan`. No rebuild, rebind, larger
arena, per-token prefill or long-prefill graph capture was needed. The library
hashes are in the preceding [ownership milestone](../2026-10-18-owned-opapi/artifact-hashes.json).

Independent early/middle/late retrieval references each have 65,520 prompt
tokens and five output tokens through first EOS, within a 65,536-token context.
The native test interleaves arithmetic peers across three residents and checks
39/39 output tokens. The HTTP frontend is vLLM Rust 0.25.1, with two active
residents and one prefix snapshot slot. It passes SSE/collected agreement,
queued isolation, concurrent short progress, cancellation recovery, context
rejection and prefix-cache salt isolation/eviction.

## Measurements

| HTTP mixed workload | Long TTFT (s) | Concurrent short TTFT (s) |
| --- | ---: | ---: |
| 64K early reference | 30.270 | 2.283 |
| 64K middle reference | 29.749 | 2.052 |
| 64K late reference | 30.775 | 2.095 |

The short request completed before the long request in every case. Cancellation
while the peer performed prefill and subsequent recovery passed.

| HTTP prefix-cache scenario | Cached tokens | TTFT (s) |
| --- | ---: | ---: |
| Cold, salt A | 0 | 43.904 |
| Warm, salt A | 61,440 | 3.858 |
| Cold, salt B | 0 | 32.747 |
| Warm, salt B | 61,440 | 2.990 |
| Salt A after eviction | 0 | 35.611 |

The separate native run measures **23.394 ms median per pure-decode step** for
one sequence (four batches, range 22.981–27.933 ms). This includes sampling,
rank consensus and the output completion boundary; mixed prefill calls are
excluded. This run has no pure two-/three-sequence decode samples, so it does
not establish their TPOT. HTTP text chunks are not token-level timing samples.
These few measurements are observations, not production SLOs or a controlled
performance comparison against omniinfer.

## Execution and resource evidence

The first native long request executes 15 × 4096 + 4080 uncaptured tokens.
Other mixed requests share the 4096-token budget with peers and therefore have
smaller chunks and additional forwards. The complete per-request geometry is
retained in `summary.json` and checked for contiguous positions, actual row
counts, sample boundaries and total batch budget. A cache hit computes only
the 4080-token remaining suffix. No prefill graph replay occurs on any rank.

All eight ranks retain arena **40,492,359,704 bytes** and shared workspace
**159,221,248 bytes**. Both runs pass zero workspace growth, zero execution
allocation/internal completion boundary, and zero activation-host transfer or
rejected transfer. Decode graphs remain enabled.

HTTP shutdown audits show 38 admissions, four mixed ticks and 405,921 newly
executed input tokens, all accounted for by completed arithmetic histories.
Two prefix hits restore 122,880 tokens, with six publications. Final snapshot
ownership and shared-page occupancy are zero; native workers close normally.
The explicit state-copy path performs 1600 D2D copies and 6,952,009,728 bytes per
rank across this complete workload. These are declared snapshot transfers.

This qualifies the new backend for these long-context correctness and cache
scenarios. It **does not close** the separate host RSS stability gate, larger
cache budgets, stochastic sampling, slow-transport behavior, fault recovery or
broader quality/performance qualification. The experimental context opt-in remains.

## Reproduction

Use `/data/p00603624/inferfabric/openpangu92` inside Docker `inferfabric-92b`
on `7.156.99.58`, with the normal CANN and both vendor environments. Rust
artifacts are the existing locally cross-compiled frontend/compiler.

* HTTP: `scripts/qualify-program-http.py`, server
  `/data/p00603624/inferfabric/inferfabric-server-apc-device-zero`, new backend
  and frozen plan above, `batched16-preparation-reference.json`, resident cases
  `retrieval-64k-reference.json`, prefix cases `device-zero-apc-prefix64-case.json`,
  two active sequences, 4096 batch tokens, `--expect-prefill packed --respect-eos
  --check-arithmetic-history --request-timeout-seconds 300`; output
  `owned-opapi64k-http`, HCCL port 40800.
* Native: generate `owned-opapi64k-fixture` using
  `scripts/prepare-sequence-fixture.py` from the new frozen plan, the 64K reference
  and arithmetic peer. Then run `scripts/qualify-sequence-serving.py --frozen-only
  --batch-trace` with the new probe, generated requests/bindings/reference and
  output prefix `owned-opapi64k-native`, HCCL port 41000.

`results.json` is the final HTTP result. The `*-progress.json` files are
intermediate checkpoints; their `complete:false` flags do not override the
final result and normal drain. Raw outputs, per-rank audits and fixture
provenance accompany the consolidated summary.
