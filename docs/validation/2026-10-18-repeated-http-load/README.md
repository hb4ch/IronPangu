# Repeated mixed HTTP load and overload — 2026-10-18

The existing qualified frontend/backend/plan completed 12 concurrent request
waves with four clients per wave and two active native residents. Each wave
included an 8176-token retrieval prompt and shorter independent references,
followed by a deliberate stream disconnect and a reference-checked recovery
request. The 60 completed wave/recovery requests matched their references
through EOS; all usage counters were checked. The repeated-load phase took
77.96 seconds. This is a short repeated-load gate, not an hours-long soak test.

A separate simultaneous burst submitted 16 long requests: eight were accepted
and eight returned HTTP 503, `server_overloaded`, with `Retry-After: 1`. Accepted
streams were deliberately disconnected, and a fresh request then returned the
correct reference output. The overload probe now honors the requested EOS scope
instead of forcing post-EOS generation.

Including baseline and mixed-progress checks, the run recorded 101 admissions,
27 mixed ticks and 133,151 executed input tokens. All executed tokens were
accounted for by arithmetic histories; a request cancelled before execution need
not have a history. Shared-page ownership returned to zero on graceful shutdown.

Every rank retained arena 40,492,359,704 bytes and workspace 159,221,248 bytes,
with no workspace growth, execution allocation, internal completion boundary or
activation-host transfer. Decode graphs remain enabled; prefill remains
uncaptured. This audit does not measure every host allocation or HTTP transport
buffer, and does not establish slow-socket memory bounds or long-duration RSS
stability.

Reproduction adds `--load-rounds 12 --load-concurrency 4 --check-overload` to the
normal `qualify-program-http.py` invocation with independent `--cases` and
`--resident-cases`, `--respect-eos`, `--check-arithmetic-history`, two sequences
and a 4096-token budget. Partial waves are retained in `load-progress.json` on
failure. Remote output is `repeated-load-http` under
`/data/p00603624/inferfabric/openpangu92`; runner exited 0, HCCL port 64200.
It used `device-zero-apc8k-frozen-plan`, `native-build-device-zero` and
`inferfabric-server-apc-device-zero`; prefix caching was disabled for this gate.
