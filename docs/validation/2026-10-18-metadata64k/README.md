# Completion-scoped metadata: 64K serving qualification — 2026-10-18

This qualification uses native revision `dfe4d7f`, library
`native-build-metadata-lease/libinferfabric_program.so` (SHA256
`b16e406816734bf2289ecc8c396e75d12cf0cfc10bcf826c4e8c6478acbc089d`),
and `metadata-lease8k-frozen-plan`. The frontend remains the locally
cross-compiled vLLM Rust 0.25.1 artifact. Native execution runs inside Docker
`inferfabric-92b` on 7.156.99.58. No rebind or arena enlargement is needed.

## HTTP output and scheduling

The HTTP runner exits 0. Independent early/middle/late retrieval references
contain 65,520 prompt tokens and five output tokens through first EOS within
the 65,536-token context. Collected/SSE agreement, queued isolation, cancellation
recovery, concurrent short-request progress and context overflow rejection pass.
The configuration is two active sequences, one snapshot slot and 4096 batch tokens.

| Mixed workload | Long TTFT (s) | Short TTFT (s) |
| --- | ---: | ---: |
| Early retrieval | 31.769 | 2.113 |
| Middle retrieval | 28.554 | 1.744 |
| Late retrieval | 26.686 | 1.571 |

All three short requests finish before their long peer. Their subsequent HTTP
text-chunk intervals nevertheless range from 1.38 to 2.00 seconds under competing
4K prefill. These intervals are not token-level TPOT; they expose an unresolved
mixed-workload latency cost. Idle decode measurements must not hide this cost.

| Prefix scenario | Cached tokens | TTFT (s) |
| --- | ---: | ---: |
| Cold A | 0 | 25.609 |
| Warm A | 61,440 | 2.935 |
| Cold B (different salt) | 0 | 27.611 |
| Warm B | 61,440 | 3.293 |
| A after eviction | 0 | 26.581 |

The hit computes the remaining 4080-token suffix. All ranks report zero prefill
graph replays. Shutdown records 38 admissions, four mixed ticks and 405,921
executed input tokens, all accounted for by arithmetic histories. Two hits
restore 122,880 tokens; six snapshots are published. Snapshot ownership and
shared-page occupancy both reach zero, and native workers close normally.

## Resource contract

All eight ranks preserve arena 40,492,359,704 bytes and shared workspace
159,221,248 bytes, with no workspace growth, execution allocation, internal
completion boundary or activation-host transfer. Final audits retain 9,918
ordinary and 516 strided permanent descriptors; transient cache entries,
invocations and live transient-origin descriptors are zero. Explicit snapshot
transfers total 1600 D2D copies / 6,952,009,728 bytes per rank across this workload.

This complements the [24-wave RSS gate](../2026-10-18-metadata-leases/README.md).
It does not establish a production soak bound, slow-socket robustness, fault
recovery, stochastic sampling or broader model quality. The experimental context
opt-in remains. Timings are observations, not controlled omniinfer comparisons.

## Reproduction and evidence

Use the normal CANN/vendor environments in
`/data/p00603624/inferfabric/openpangu92`. Run `qualify-program-http.py` with
`--plan metadata-lease8k-frozen-plan`, the library above, server
`/data/p00603624/inferfabric/inferfabric-server-apc-device-zero`,
`--cases batched16-preparation-reference.json`,
`--resident-cases retrieval-64k-reference.json`,
`--prefix-cases device-zero-apc-prefix64-case.json`,
`--max-num-seqs 2 --max-num-batched-tokens 4096 --expect-prefill packed
--respect-eos --check-arithmetic-history --request-timeout-seconds 300`.
Output is `metadata-lease64k-http`, HTTP port 18085, HCCL base port 41800.

`results.json` plus normal drain and exit 0 establish the HTTP result.
`*-progress.json` files are intermediate checkpoints, even when the full run
passes. Raw results, histories, transfers and per-rank audits are retained here.

## Native geometry and TPOT

The frozen native runner also exits 0: six interleaved requests across three
residents match all 39 reference tokens through EOS. Its 53 completed batches
account for 196,629 input tokens. The first long request uses 15 × 4096 + 4080
uncaptured tokens. Other long requests share the batch budget with short peers;
the checker verifies actual tails, contiguous histories and full budget use.
No per-token prefill or long-prefill graph is introduced.

Four pure single-sequence decode batches have median **23.6015 ms**, range
23.452–25.037 ms. This includes the completed native ABI call, sampling and rank
consensus. Competing-prefill batches are excluded. There are no pure two-/three-
sequence samples in this run: this is a small single-sequence observation, not
multi-sequence TPOT or an end-user latency guarantee. Decode graphs replay and
all eight ranks pass the same device-residency and memory invariants above.

Generate `metadata-lease64k-fixture` with `prepare-sequence-fixture.py`, the new
frozen plan, `retrieval-64k-reference.json` and the arithmetic peer from
`batched16-preparation-reference.json`. Run `qualify-sequence-serving.py
--frozen-only --batch-trace` with `native-build-metadata-lease/program_engine_probe`
and the generated requests/bindings/reference; output prefix
`metadata-lease64k-native`, HCCL port 42000. Fixture provenance, complete trace,
output tokens and per-rank audits accompany `summary.json`.
