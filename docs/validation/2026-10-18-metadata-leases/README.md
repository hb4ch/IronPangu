# Completion-scoped tensor metadata — 2026-10-18

The executor now gives descriptor metadata an explicit lifetime independent of
the device-buffer lease. An uncaptured forward opens an epoch that continues
through sampling and rank consensus. At the existing stream completion boundary,
transient invocations are destroyed before ordinary/strided descriptor caches
are cleared. This adds no synchronization, device copy or device allocation.

`DeviceTensor` and resident aliases/views retain shared descriptor owners as
well as their existing buffer leases. A value can therefore outlive the cache
entry safely. Immutable weight-layout cache keys explicitly use the permanent
strided cache; decode/graph descriptors and invocations also remain permanent.
Capture rejects an unfinished uncaptured metadata epoch, even after the
forward's lexical `BoundedBuffers` scope has returned.

## Attribution before this change

`owned-pool-stack-http` repeated eight four-client load waves on the preceding
owned-opapi backend. Forty wave/recovery outputs matched through EOS, with
normal drain, 75 admissions, 20 mixed ticks and zero final shared-page ownership.
RSS grew 155,754,496 bytes, exceeding the unchanged 128 MiB gate (exit 1).

Acknowledged between-wave snapshots show unscaled sampled live-byte growth of
187,087 for libc and 30,272 for CANN pool clients. These ownership levels overlap
and must not be added or extrapolated into an exact heap measurement. All
positive pool-client growth is in the native descriptor/view families, resolving
to `Context::descriptor`, `Context::strided_descriptor`, `Tensor::view`,
`Decoder::pre` and cache-append control views. The earlier vendor placeholder
stacks are absent from these positive-growth samples. The largest libc families
include HCCL `TaskExceptionHandler::InsertTaskMap`, whose bounded history remains
a separate source of retained host memory.

The `attribution` directory preserves raw compressed snapshots, wave timestamps,
grouped stacks, installed-library hashes/symbols, failed resource gate and drain
log. Sampling is evidence for targeting this change, not proof that descriptor
caches explain all RSS growth.

## Verification

The Docker `prepared_execution_probe` passes the existing ordered-buffer,
prepared-invocation, strided-boundary, transfer and arena contracts. New cases
execute sampling outside the forward's lexical scope, check its output, reject
capture before completion, and verify descriptor expiration using weak owners.
An escaped tensor stays valid across completion; its descriptor expires after
the final owner is released. The immutable weight-view cache remains valid.
No device reset is used. Twelve HTTP helper tests also pass.

All three full 92B stages (discovery, fixed workspace, frozen arena) pass
39 exact output tokens over six interleaved requests. Decode graphs replay;
every rank has zero activation-host transfers, execution-time device allocations
or internal completion boundaries. Frozen arena and workspace sizes remain
40,492,359,704 and 159,221,248 bytes per rank respectively.

Each frozen rank retains 9,918 ordinary and 516 strided permanent descriptors.
Transient cache entries, live transient-origin descriptors and transient
invocations are all zero at final audit. Peak live transient descriptors are
2,545 per rank. The audit intentionally distinguishes cache entries from actual
live leases: clearing a cache does not necessarily destroy a still-owned tensor.

Build/source: `native-build-metadata-lease` / `native-metadata-lease-source` in
`/data/p00603624/inferfabric/openpangu92`, Docker `inferfabric-92b` on
`7.156.99.58`. Qualification uses unbound `owned-opapi-source`, the existing
`device-zero-apc8k-fixture`, and fresh prefix `metadata-lease8k` (HCCL port 41400).
`qualification.json` records all stages, per-rank metadata counts and library
hash. No older backend's frozen arena is reused.

The 24-wave, four-client HTTP gate passes with 120 correct EOS-scoped load and recovery outputs. Post-wave RSS grows 25,329,664 bytes (24.16 MiB), below the unchanged 128 MiB limit; threads remain 631 and file descriptors 76. Normal drain records 171 admissions, 47 mixed ticks, 237,111 consumed tokens and zero remaining shared-page ownership. All eight final audits retain 9,918 ordinary descriptors, 516 strided descriptors and 1,017 invocations, with zero transient entries or live transient descriptors. Raw evidence is in `metadata-lease-load-http/`.

This closes the measured repeated-load regression, not the longer-duration production soak gate. The previous library's 64K timings do not qualify this revision.
