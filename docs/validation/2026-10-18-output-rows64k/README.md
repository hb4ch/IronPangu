# Sampled-row backend: 64K HTTP qualification — 2026-10-18

The HTTP qualifier exits 0 on `native-build-output-rows` and
`output-rows8k-frozen-plan`, through the locally cross-compiled vLLM Rust 0.25.1
frontend. Three independent early/middle/late retrieval cases each contain
65,520 prompt tokens and five outputs through first EOS. Collected/SSE agreement,
queued isolation, cancellation recovery, concurrent short progress, context
rejection, prefix salt isolation and eviction all pass.

## Observed latency (not an improvement claim)

| Mixed retrieval | Long TTFT (s) | Short TTFT (s) |
| --- | ---: | ---: |
| Early | 33.985 | 1.313 |
| Middle | 49.693 | 3.053 |
| Late | 38.038 | 3.080 |

Short requests finish before their long peers, but their later text-chunk intervals
range from 1.07 to 2.86 seconds. Text-chunk intervals are not token-level TPOT.
The preceding metadata-only backend observed 26.69–31.77-second mixed long TTFT.
These are separate runs, not a controlled alternating A/B test; nevertheless this
run is slower and supplies no evidence of latency improvement. Retain all samples.
Do not promote sampled-row execution on a performance claim or disable the compile
opt-in before the remaining investigation. The reduction in arena reservation is
independently established.

| Prefix scenario | Cached tokens | TTFT (s) |
| --- | ---: | ---: |
| Cold A | 0 | 26.235 |
| Warm A | 61,440 | 2.824 |
| Cold B | 0 | 25.567 |
| Warm B | 61,440 | 2.887 |
| A after eviction | 0 | 30.886 |

## Execution and resource checks

All ranks retain the 38,009,938,432-byte arena and 159,221,248-byte shared workspace.
The qualifier checks no execution allocation, internal completion boundary,
workspace growth or activation-host transfer. Transient descriptor caches,
invocations and live transient descriptors drain to zero. Prefill graph replay
counts are zero; decode graphs stay enabled.

Shutdown records 38 admissions, four mixed ticks and 405,921 executed input tokens,
all covered by arithmetic histories. Two cache hits restore 122,880 tokens and
six snapshots are published. Shared pages and snapshot ownership return to zero;
workers close normally. Declared snapshot D2D transfers remain 1600 copies and
6,952,009,728 bytes per rank across the complete workload.

## Artifacts and reproduction

Library SHA256 `584ed7e498833498438553126fd9308a7cf5070a540ff5acd8604cde894b2ecf`;
execution SHA256 `4260ad065d0ab512b5ac964b7b5b27bd3ea96e69ac7aa4aaf044c30b2c5b1bf8`.
Docker `inferfabric-92b`, workspace `/data/p00603624/inferfabric/openpangu92`.
Use `qualify-program-http.py` with the new library/frozen plan, existing
`batched16-preparation-reference.json`, `retrieval-64k-reference.json`,
`device-zero-apc-prefix64-case.json`, two active sequences, 4096 batch tokens,
`--expect-prefill packed --respect-eos --check-arithmetic-history
--request-timeout-seconds 300`, HTTP port 18085 and HCCL base port 42600.
Output directory: `output-rows64k-http`.

Raw final results, intermediate progress, arithmetic histories, shutdown log and
per-rank audits are retained. Final results and exit 0 establish completion;
progress files alone do not. Native 64K batch timing and the post-change collective
capture are a separate ongoing qualification, not established by this HTTP pass.
Production soak, slow-socket behavior, fault recovery and broader quality remain
release gaps; experimental context/output-row gates remain in place.
