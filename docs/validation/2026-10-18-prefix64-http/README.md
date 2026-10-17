# 64K prefix restoration through HTTP — 2026-10-18

The current three-resident frozen plan passed the full HTTP cache gate with a
65,520-token independent retrieval reference, two active slots and one snapshot
slot. Cold requests use sixteen uncaptured forwards (15 × 4096 + 4080). Warm
requests restore 61,440 tokens and compute the remaining 4080-token suffix.

| Request | Cached tokens | TTFT (s) |
| --- | ---: | ---: |
| Cold, salt A | 0 | 33.062 |
| Warm, salt A | 61440 | 2.974 |
| Cold, salt B | 0 | 33.884 |
| Warm, salt B | 61440 | 3.678 |
| Salt A after eviction | 0 | 31.876 |

All outputs matched through EOS, including warm continuations and eviction.
Mixed 8K/short requests, cancellation recovery, queued isolation and context
rejection also passed. Measurements are sequential requests in one process,
not an SLO or a repeated controlled latency distribution. The cache metric and
arithmetic audit account for the skipped prefix; no prefill graph was captured.

The run recorded 38 admissions, eight mixed ticks, two cache hits, 122,880
restored tokens and six publications. Arithmetic histories account for all
233,891 newly executed input tokens. Every snapshot and shared page was reclaimed
at shutdown. Each rank executed 1600 explicit D2D state copies totalling
4,912,705,536 bytes across the complete run.

All eight ranks retained arena 40,492,359,704 bytes and workspace 159,221,248
bytes, with no growth, execution allocation, internal completion boundary or
activation-host transfer. Decode graphs remain enabled. This qualifies one
snapshot slot and the tested reference workloads; larger cache budgets,
transport backpressure, sustained load and broader quality remain release gates.

Runner exited 0; HCCL port 63800. Remote evidence is `device-zero-apc64-http`
under `/data/p00603624/inferfabric/openpangu92`, using
`device-zero-apc8k-frozen-plan`, `native-build-device-zero` and
`inferfabric-server-apc-device-zero` behind vLLM Rust 0.25.1.
