# Retryable admission overload in the vLLM Rust frontend

2026-10-17. A full native admission queue now produces typed `Overloaded` errors;
a stopped/unhealthy backend produces typed `Unavailable` errors. Completion and
chat submission map these to HTTP **503**, with codes `server_overloaded` and
`backend_unavailable`, respectively, and `Retry-After: 1`. This is a retry hint,
not a promise that capacity returns within one second. Ordinary execution errors
remain 500 and invalid requests remain 400. Existing post-header SSE error
handling is unchanged.

The adapter distinguishes full and closed channels instead of collapsing both
into `native engine stopped or busy`. Rejected jobs never enter the active-ID
map; a rejected ID can be retried after capacity becomes available. Queue bounds
and device execution are unchanged. The adaptation remains in the pinned vLLM
0.25.1 Rust frontend; no alternate HTTP server or Python serving path is added.

## Local verification

- 305 vLLM server library tests pass, including completion/chat classification,
  status/header snapshots, existing HTTP routes and streaming behavior.
- 17 InferFabric frontend tests pass, including deterministic queue-full,
  channel-closed, unhealthy-backend, cancellation and same-ID retry checks.
- Rust frontend cross-compiles locally in WSL with the locked dependency graph.

The first broad upstream run stalled on a localhost HTTP test while proxy
variables were inherited. The isolated test and then all 305 tests pass with
proxy variables cleared and `NO_PROXY=127.0.0.1,localhost`. This is a test-process
environment change, not a persisted proxy configuration change. Snapshot updates
use `CARGO_WORKSPACE_DIR` pointing at `vendor/rust` for its nested workspace.

## Real NPU HTTP verification

TP8/EP8 OpenPangu 92B, two residents, sixteen-token prefill, token budget 32,
context 4,096, using the existing batched-rotary frozen executable. The qualifier
passes standard full text/SSE fixtures, six concurrent requests, cancellation
during a 4,088-token peer prefill, short-peer progress, validation rejection and
graceful shutdown before/alongside the added overload gate.

The new `--check-overload` gate starts sixteen simultaneous long streaming
requests, retaining accepted connections until all response headers are observed:

- Eight requests are accepted into the bounded queue.
- Eight receive 503, `server_overloaded` and `Retry-After: 1`.
- Accepted burst streams are disconnected before resident admission; a subsequent
  ordinary request returns the exact expected output.
- Final shared-page occupancy is zero; eighteen total resident admissions include
  the normal regression and the recovery request. Rejected/queued-cancelled burst
  requests do not consume resident pages.
- All eight rank audits retain frozen arena, planned workspace, zero growth,
  zero activation host transfers/rejections, and zero retained step allocations
  or internal fences.

See [results.json](results.json), [rank-audits.json](rank-audits.json) and
[server.log](server.log). This verifies queue overload and recovery, not a sustained
load/slow-client/rank-failure soak. Native backend-unavailable response mapping is
covered by local tests; no destructive NPU rank-failure experiment was added.

## Reproduction

Run `scripts/qualify-program-http.py` with the ordinary batched-rotary configuration
and `--check-overload --max-num-seqs 2 --max-num-batched-tokens 32
--resident-cases batched16-long-reference.json`. Keep the long native timing run
separate from this HTTP workload.

- Frontend SHA256: `a5488146736631b1690937d6797c81bbd52ce189babdc878f86cd0bffe662680`.
- Native SHA256: `581d123c08158ed1993d6decdab34603356bb612b6ec484f3617c03228091ad4`.
- Execution SHA256: `fa56a54d2c5623044a52e97592e9a098ce8af5aaad5d1d29e27f4ebb2622fa53`.
- Remote artifact: `/data/p00603624/inferfabric/openpangu92/overload-http` in
  Docker `inferfabric-92b` on `7.156.99.58`; binary `../inferfabric-server-overload`.
- Local raw evidence: `.deploy/overload-evidence`; compact audits retain raw hashes.

The 64K, numerical parity, prefix caching, multi-sequence kernels and broader
production readiness gates remain open.
