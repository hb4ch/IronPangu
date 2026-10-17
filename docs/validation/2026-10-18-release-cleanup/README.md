# Safe resident cleanup after audit failure — 2026-10-18

After a successful native resident reset, audit I/O failure no longer bypasses
host ownership cleanup. The runtime retains the audit result, clears completed
history/import bookkeeping and retires the idle lease, then returns the original
audit error. Native reset failures still poison shared state and return before
this cleanup; potentially in-flight pages are not forcibly retired. The change
adds no device call, fence, transfer or fallback.

The strengthened full-92B HTTP fault test passes on the new frontend:

- A normal request matches independent EOS-scoped reference text and usage.
- The next exclusive audit write is deliberately collided inside the test's own
  directory. The SSE response contains the expected `File exists (os error 17)`
  error and no successful terminal choice.
- Health and subsequent admission both return 503; the process remains available
  to report failure and drains normally on SIGTERM.
- Fault-stage logical shared-page occupancy is **zero**, versus 15,458,304 bytes
  in the original failure evidence. The error remains visible to the client.
- A fresh process opens the same checkpoint, frozen plan and native library,
  matches the reference output again and drains with zero logical occupancy.
  Both processes exit zero. Readiness takes 84.0 / 87.0 seconds in this run;
  these timings are observations, not a startup SLO.

This qualifies one release-time host error and explicit process restart. It does
not qualify hardware rank crashes, failure broadcast across multiple concurrent
requests, automatic restart/failover or long-duration failure churn. No checkpoint
or plan is modified and no device reset is used. Raw stages, injected file,
responses, logs and artifact hashes are preserved in
`release-failure-cleanup-http/`; its `results.json` is the completed report.

## Verification and reproduction

Ten native library unit tests, 20 program-worker tests and two fault-checker tests
pass. The frontend ARM binary is cross-built locally with `cross-frontend.sh`;
strict frontend Clippy passes. Strict native-package Clippy still reports existing
`collapsible_if` and `manual_is_multiple_of` findings at program.rs:149 and :193;
the broader workspace invocation also encounters pre-existing compiler lints.
Those checks are not claimed green. The tested runtime fix changes only the
successful-reset cleanup tail.

Docker `inferfabric-92b`, host 7.156.99.58, workspace
`/data/p00603624/inferfabric/openpangu92`, normal CANN/vendor environment, HOST
expansion and explicit decode-graph/long-context flags. Execute
`qualify-program-release-failure.py` with server
`../inferfabric-server-audit-cleanup`, plan `bulk-domains8k-frozen-plan`, checkpoint
`/models/92B`, library `native-build-bulk-domains/libinferfabric_program.so`, cases
`batched16-preparation-reference.json`, output `release-failure-cleanup-http`,
HTTP port 18085, HCCL base port 45600 (restart uses 45800).

Frontend SHA256:
`816c355421c27c985801119fe113c01f5fdd1a3b567f4fe0b1b91fdc1e778a4b`.
Native library and execution hashes remain those of the qualified bulk-domain
baseline. Long-context and prefix performance reports on the preceding frontend
remain historical evidence; this narrow failure fix is not a new 64K performance
measurement.
