# Device-zero 64K HTTP qualification — 2026-10-18

vLLM Rust 0.25.1 completed the full HTTP qualification using
`device-zero4k-frozen-plan` and `native-build-device-zero/libinferfabric_program.so`.
The test used two active sequences, a 4096-token budget, uncaptured prefill and
native decode graphs. Prefix caching was disabled. Each long prompt had 65,520
tokens. Outputs matched independent references through first EOS.

| Retrieval placement | Previous HTTP TTFT (s) | Device-zero HTTP TTFT (s) |
| --- | ---: | ---: |
| Early | 97.718 | 61.226 |
| Middle | 99.877 | 42.015 |
| Late | 99.575 | 33.810 |

These are sequential requests in one server process, not repeated controlled
trials. The range is substantial; neither the fastest value nor the average is
a production SLO. The previous reference is `../2026-10-18-http-64k`. The new
server includes lazy expert descriptors as well as planned device zero.

All 27 admissions completed the isolation/recovery checks. Eight concurrent
queued requests, cancellation during peer prefill, short-request progress,
context-overflow rejection and graceful shutdown passed. All three short peers
emitted before their long peer. Their output intervals still include seconds of
prefill interference; decode latency under mixed load remains an optimization
target. SSE chunk intervals are not claimed as per-token TPOT.

The scheduler accounted for 201,181 input tokens, reached two active requests
and recorded four mixed ticks. All shared pages were reclaimed. Every rank
retained its 39,572,690,960-byte arena and 159,221,248-byte workspace, with no
growth, in-execution allocation, internal completion boundary or activation
host transfer. Prefill graph replay count was zero on all ranks.

Remote output: `/data/p00603624/inferfabric/openpangu92/device-zero64k-http`.
Runner exited 0; HCCL port 63200, HTTP port 18085. The request socket timeout was
300 seconds. Raw results and independently checked resource counters accompany
this note. This qualifies long-context correctness for this regression suite;
prefix reuse, sustained-load/fault coverage and broader model quality remain
separate requirements.
