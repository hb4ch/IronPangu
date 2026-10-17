# FD lifecycle diagnostics and 64K rerun — 2026-10-18

The original one-FD excursion is **not attributed**. Its failed evidence remains
in `../2026-10-18-bulk-domain-load/`; do not replace it with this passing rerun.

The same 24-wave 64K workload, with FD target/TCP-state recording and unchanged
128 MiB RSS / zero thread-or-FD growth gates, passes on the unchanged frontend and
v9 native backend. All 120 load/recovery reference outputs match through first
EOS. RSS grows 71.465 MiB; threads remain 663 and FDs remain 76 at all post-wave
samples. Final native audits, decode replay, device residency and frozen-resource
checks pass; 171 admissions and 1,785,397 input tokens are accounted for, shared
pages drain to zero, and graceful shutdown completes. No long-prefill graph is
replayed. This finite run is not a production soak claim.

`bulk-domains64k-fd-load-http/` preserves the terminal report and all raw evidence.
Reproduction matches the prior workload, with the FD-instrumented qualifier,
HCCL base port 44600 and output prefix `bulk-domains64k-fd-load-http`. The frontend
remains `inferfabric-server-apc-device-zero` and native plan/library remain the
qualified bulk-domain versions. No serving fix was applied during this rerun.

## Independent connection diagnostic

`scripts/diagnose-http-fd-lifecycle.py` runs the same vLLM Rust HTTP listener with
its existing synthetic mock backend, without model weights or NPU occupation.
It probes either health requests or SSE chat completions, closing at `[DONE]`
just like the serving qualifier. It records FD identities immediately after each
four-client wave and, if elevated, bounded cleanup samples for up to two seconds.
These are diagnostic observations, not retries that turn a resource gate green.

Both health and SSE runs complete 200 four-client waves (800 measured requests
plus warmup each) without FD growth; both processes exit zero after graceful
shutdown. This does not explain the original excursion or qualify slow-client
backpressure. The diagnostics ran on port 18086 while the independent NPU
resource run was active; its latency results must not be used for a controlled
performance comparison.

`health-summary.json` and `sse-summary.json` retain per-wave counts, baseline/final
identities and SHA256 of the raw observation files. Raw files remain in remote
`http-fd-diagnostic/observations.json` and `http-sse-fd-diagnostic/observations.json`
under `/data/p00603624/inferfabric/openpangu92`. Reproduce with the existing
`frontend/fixtures/mock.inferfabric` and `frontend/fixtures/mock-tokenizer`, the
same server executable, `--rounds 200 --concurrency 4 --endpoint health|sse`.

## Completion ordering found during inspection

The current worker publishes terminal output before `release_resident` completes.
That release can write an arithmetic-audit file and fail. Thus an HTTP success can
precede resource cleanup or its error. This is an independently established
ordering issue, not proof of the historical FD's identity. A candidate fix holds
only the terminal output until successful release and active-request removal;
release failure produces an error instead. Local frontend tests (31 passed, one
NPU-only ignored), strict Clippy and local ARM cross-build pass. Full NPU
qualification of that candidate is separate and still in progress.
