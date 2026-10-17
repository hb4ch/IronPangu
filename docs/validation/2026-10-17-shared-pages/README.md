# Shared page ownership in native HTTP serving

Date: 2026-10-17. OpenPangu V2 92B, TP8/EP8, two resident histories, four-token
prefill, C8 DSA and BF16 SWA caches. Context remains capped at 512.

## Compiled and runtime contract

`compile-token --shared-pages --resident-sequences 2 --prefill-tokens 4` emits
`IFTOKN05`, with the shared policy recorded in the serving manifest and viewer.
The physical pool size is unchanged: each cache component/rank has the existing
resident-count times per-request capacity, allocated before graph capture.
This is dynamic ownership within preallocated HBM, not a reduction of reserved
HBM or support for a separately configurable pool budget.

At admission Rust reserves all component/rank resources atomically for the
request's prompt-plus-output limit. Recurrent slots are pinned to the selected
resident ID; cache pages come from shared free lists. Native bindings use the
compiled logical state ID, not an inferred model-specific name. Partial and
noncontiguous tables are legal; duplicate pages, out-of-range pages and pages
owned by another resident are rejected before ownership changes.

Each native invocation is registered as an allocator flight. The synchronous
ABI returns after the owning streams on every rank complete, after which Rust
records completion. Completion/cancellation clears only the resident's owned
cache pages and recurrent slot before retiring the lease. Rank failure poisons
the pool rather than pretending work completed. Startup releases temporary
preparation partitions before accepting serving reservations. Graph addresses,
weights and the maximum workspace remain stable.

This does not yet implement shared read-only prefix pages, copy-on-write,
SWA ring reclamation or prefix snapshot/restore. Existing allocator borrower
semantics alone do not establish a complete model-state checkpoint.

## Evidence

- [Native fixtures](native-results.json): all 99 expected output tokens match.
- [Boundary continuations](boundaries-results.json): all 336 serial-oracle output
  tokens match across 42 interleaved requests, lengths 1–504, shuffled page maps.
  These probes exercise the new policy with complete per-request page maps;
  dynamic partial reservations are exercised by the HTTP run below.
- [HTTP results](http-results.json): full collected/SSE text and usage match;
  six concurrent requests, cancellation with a live peer, replacement admission,
  rejection of unsupported sampling/context overflow, and graceful shutdown pass.
- [Scheduler/ownership log](server.log): 17 admissions, **736 component/rank
  bindings outside the old fixed partition**, peak occupied ownership
  **202,866,688 bytes across eight ranks**, and **zero remaining occupied bytes**.
  This counter is ownership of the preallocated pool, not device-wide HBM usage.
- [Native](native-audit.json), [boundary](boundaries-audit.json) and
  [HTTP](http-audit.json) audits pass on all eight ranks: no activation host
  transfers, rejected transfers, retained-step allocations/internal fences or
  workspace growth. Arena remains **12,450,806,784 bytes/rank**, workspace
  **130,916,352 bytes/rank**.

In the HTTP mixed workload, the 504-token request has **4.881 s TTFT**, while the
short replacement finishes correctly in **628 ms** before the long peer completes.
This is slower than the fixed-partition HTTP result (4.663 s and 582 ms).
Per-state/rank binding commands and allocator bookkeeping add overhead; batched
binding submission and profiling remain optimization work. Isolated short-stream
text-chunk intervals are roughly 18.2–18.5 ms; these are not labeled per-token
latency because chunks can combine tokens.

Local compiler, IR, runtime and native Rust tests pass. Native Docker tests cover
partial bindings, failed-update rollback, cross-owner rejection and release/reuse.
Allocator tests cover pinned recurrent slots, all-rank rollback, stale handles,
borrowers and delayed rank retirement. Frontend tests and strict frontend Clippy
pass. Synthetic 64K ownership tests are not 64K model-output qualification.

## Reproduction and artifact identity

Use fresh workspace discovery, planned-workspace preparation and arena binding
for the `--shared-pages` program. Qualify `shared-pages-v1-bound` with
`qualify-program-engine.py --packed-prefill --interleave --permuted-pages`.
Run `qualify-program-http.py` with `--max-num-seqs 2 --max-num-batched-tokens 8
--expect-prefill packed --resident-cases resident-boundary-cases.json`.
For shared manifests, the HTTP harness requires actual cross-partition use and
zero remaining ownership, as well as all-rank transfer/memory audits.

| Artifact | SHA-256 |
|---|---|
| Execution | `1d3c456703184d977050d0c015de2a5d9fc2adc9849b71ca28ae05da5f9a0ba8` |
| Native library | `08cde26ca6bed0ecfa5683eb28072ae6d58b37895ffe6823a52aff385960e430` |
| Native probe | `7b11de8c272d3709c9687fdae30127b64ebe5a2a7624171c304c6d5321b17683` |
| Rust server | `f964e6e2c77df8806656ebc70ec7baba5a1673b2a3b3094e2a60760313746d06` |

Remote directories `shared-pages-v1-qualified`, `shared-pages-v1-boundaries` and
`shared-http-v2` retain raw audits; compact records here include audit hashes.
Native builds and execution ran in Docker; Rust builds ran locally in WSL.

## Remaining release gates

Learned DSA/index-cache execution, long-context table sizing and SWA recycling,
hybrid prefix snapshots/trie reuse, configurable pool budgets, efficient larger
chunks/batched decode, stochastic sampling and 64K reference/performance/robustness
qualification remain open. No production-readiness claim is made by this milestone.
