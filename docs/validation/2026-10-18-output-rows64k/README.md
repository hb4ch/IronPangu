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
progress files alone do not. The native/collective qualification is recorded below; HTTP and profiler timings have different scopes.
Production soak, slow-socket behavior, fault recovery and broader quality remain
release gaps; experimental context/output-row gates remain in place.

## Native 64K and post-change collective capture

The frozen native qualifier also exits 0: six interleaved requests match all
39 reference tokens through EOS, accounting for 196,629 input tokens. The first
long request executes 15 × 4096 + 4080 tokens; mixed peers share the 4096-token
budget and have additional smaller/tail chunks. All ranks preserve the arena,
workspace, device residency and decode graph contracts.

Exactly one initial 4096-token, zero-sample-row prefill batch is profiled. Its
completed ABI duration is 2,291,880 us including profiling overhead. On every
rank the exported HCCL table contains exactly **92 FP32 AllReduce calls and no
AllGather**. Each reduction has 10,485,760 input elements (40 MiB). This confirms
that the previous 148 MiB/rank vocabulary AllGather was removed from this call;
it does not remove the remaining large reductions. Rank 0 task span is 2286.42 ms.

After the profiler stops/finalizes, four pure one-sequence decode batches take
24.719, 23.145, 22.830 and 22.941 ms: median **23.043 ms**. These completed ABI
samples include sampling and rank consensus, exclude competing-prefill batches,
and occur after an earlier profiling window in the same process. There are no
pure two-/three-sequence samples. They are not a controlled speedup comparison
or a serving SLO. The slower HTTP observations above remain part of the result.

Reproduce with fresh `output-rows64k-fixture` from the frozen plan, the 64K
references and arithmetic peers. Run the new `program_engine_probe` through
`qualify-sequence-serving.py --frozen-only --batch-trace`, output prefix
`output-rows64k-native`, HCCL base port 42800, and
`INFERFABRIC_BATCH_PROFILE_DIR=/data/p00603624/inferfabric/openpangu92/output-rows4k-profile`.
Export with msprof, then use the serving timeline, kernel and collective summary
scripts. Raw CANN capture remains at that remote Docker path; exported summaries
here preserve input table hashes. Full native traces, fixture provenance and
resource audits are included. Export succeeded; cluster tuning was not complete.
