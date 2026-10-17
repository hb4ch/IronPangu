# Completed arithmetic history and immutable continuation ownership

2026-10-17. Two prerequisites for safe hybrid prefix reuse are implemented.
They do not enable cache hits or replace the remaining native snapshot/restore
and radix-index work.

The Rust native adapter records each resident's completed input intervals with
the exact numerical class and query width. It validates contiguous positions
before dispatch and appends only after every native rank completes. Reset clears
only that resident's history. Equal adjacent intervals coalesce; prefix extraction
rejects boundaries inside an invocation. Prefixes with different earlier arithmetic
remain incompatible even when their final variant matches. No tensor bytes or
prompt token IDs enter this metadata.

The generic state pool can seal an allocation after all rank work completes.
Sealed allocations reject write submissions. Read pins survive owner retirement;
storage is reclaimed only after all pins and rank completions are gone. Reused
physical page IDs do not revive stale handles. Sealing proves ownership and
completion, not that the device contains a complete model checkpoint. There is
no unseal operation and no implicit host copy.

## Full native HTTP integration

The pinned vLLM 0.25.1 Rust frontend used the existing 92B TP8/EP8 frozen family
plan, two residents, context 4,096 and token budget 32. The test passes ordinary
text/SSE reference cases, six concurrent requests, cancellation during a 4,088-token
peer prefill, short-peer progress, context/sampling rejection, overload recovery
and graceful shutdown.

The test explicitly uses normal EOS stopping (`--respect-eos`), preserving the
full references and output budget. The long request returns five completion tokens
including EOS, with correct usage. The report identifies comparison scope as
`through_first_eos` and `full_fixture_text: false`; separate forced-continuation
failures in the 32K qualification are not erased or reclassified.

All **18 completed histories account for exactly 4,462 scheduler input tokens**,
including cancelled requests. The qualifier checks every interval's declared
shape/class, contiguity, invocation alignment, context bound and unique sequence.
`completed-arithmetic.json` contains the actual records. All shared-pool allocations
are reclaimed. All ranks retain the 12,620,067,840-byte frozen arena and
131,020,288-byte workspace, with no activation host transfers, workspace growth,
replay allocations or internal completion fences.

The independent 64K reference ran on devices 8–15 concurrently. HTTP timing fields
are retained for diagnostics, **not isolated performance evidence**.

## Verification and remaining scope

Local verification passes three arithmetic-history tests (including 64K coalescing
and mixed-class rejection), nine state-pool tests (including delayed-rank sealing,
retirement, readers and stale generations), five runtime tests, four serving-IR
tests, six native Rust library tests, 18 frontend tests and 12 Python qualification
tests. Rust cross-compilation ran locally in WSL; NPU serving ran in Docker.

The real HTTP test exercises history recording and existing mutable-state lifecycle.
The new sealing path is covered by ownership tests, not a native snapshot test:
snapshot D2D operations, checkpoint-slot budgets, complete MoME/SWA/DSA/index
capture/restore, and the radix cache are still required. Native activation residency
and ACLGraph behavior are unchanged. `identities.json` pins the tested binaries
and plan; compact audits retain hashes of their complete remote source files.
