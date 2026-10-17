# Backend release error and restart: cleanup gap found — 2026-10-18

The full 92B frontend correctly propagates an injected release-time audit error:
its stream contains `File exists (os error 17)` and no successful terminal choice;
health changes to 503 and a subsequent request receives 503 `backend_unavailable`.
The process drains normally. An explicit new process with the same plan/library
then returns the independent baseline text and token usage correctly.

However, the fault-stage log records **15,458,304 bytes of logical shared-page
occupancy** at teardown. Native reset succeeds, then audit I/O fails before the
host lease is retired. This is an ownership-cleanup gap, not evidence that HBM
survives process exit. Do not claim the resource-failure gate passed.

Raw `release-failure-http/results.json` records the original narrower test scope
(error return/unhealthy rejection/explicit restart) as passed. The strengthened
qualifier now additionally requires the exact injected failure and zero logical
page occupancy in both stages. It rejects this saved fault log, as recorded in
`summary.json`; the raw original result has not been rewritten.

The injection creates only `request-2.json` in this run's otherwise empty audit
directory, after a successful baseline request has completed `request-1.json`.
The existing exclusive audit write fails deterministically. No checkpoint,
execution plan, device reset, runtime fault hook or other process is modified.
This is not a hardware rank crash, automatic recovery or multi-request failure
broadcast qualification.

`scripts/qualify-program-release-failure.py` is reusable: it performs baseline,
injection, structured-error/no-success checks, unhealthy health and admission
checks, shutdown, explicit process restart and reference recovery. Its two local
helper tests reject success followed by an error, unrelated errors and missing or
nonzero pool-drain evidence. The host ownership cleanup candidate preserves the
audit error but retires idle pages after a successful native reset; its full NPU
rerun is pending.

Reproduce inside `inferfabric-92b` on 7.156.99.58, workspace
`/data/p00603624/inferfabric/openpangu92`, with normal CANN/vendor environment,
HOST expansion and decode-graph/long-context flags. Original server:
`../inferfabric-server-release-order`; plan `bulk-domains8k-frozen-plan`, library
`native-build-bulk-domains/libinferfabric_program.so`, checkpoint `/models/92B`,
cases `batched16-preparation-reference.json`, HTTP port 18085, HCCL base port 45200
(restart uses 45400), output `release-failure-http`. The strengthened qualifier
will fail this original version's logical-resource gate. Original artifact hashes
are included in the raw final report.
