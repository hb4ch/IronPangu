# Device cache page bindings — 2026-10-17

BF16 and packed C8 cache writes now translate logical positions through the same
component/rank page mapping consumed by attention. The persistent native ABI can
bind page tables after request reset without moving cache data or graph addresses.
The Rust native adapter exposes that operation. Negative, duplicate, out-of-range
and incorrect-length page bindings fail closed; unsupported row-quantized INT8/4
bindings are rejected. Logical position remains separate for RoPE and causal masks.

The full 92B TP8/EP8 C8 program passed nine alternating requests with all **99
fixture tokens unchanged**. Each of the 46 caches on each of eight ranks uses a
seeded independently shuffled page permutation, recorded in `page-bindings.txt`.
The test resolves logical state IDs through `device-program.json` source mapping;
it does not construct physical instruction names from naming conventions.

Discovery and fresh fixed-workspace preparation also passed the 99 tokens with
identity mappings. Bound execution retains ACLGraph, zero workspace growth and
zero activation host transfers. Arena capacity is **12,408,118,336 bytes/rank**;
workspace remains **130,916,352 bytes/rank**. `rank-audits.json` includes raw audit
hashes and final device-wide HBM snapshots. Per-step records cover the final
request; graph replay counts and transfer audits cover the engine lifetime.

Measured short-prompt TPOT is **20.16–20.50 ms** per request, slower than the
preceding approximately 17.9 ms serial baseline. Additional per-component control
uploads are a known cost to consolidate in batch metadata staging. Engine TTFT is
300–368 ms and includes reset and all page-table bindings; this is not a matched
HTTP comparison. Raw token timings and output IDs are retained.

Local compiler tests verify that every cache state resolves to exactly one native
instruction and that viewer recipes expose device block tables and physical
append positions with the correct page size. Native portable tests pass inside
Docker for fragmented address translation and invalid-update rollback. The Rust
native tests pass. Compiler cross-compilation remains local.

Reproduce using `scripts/qualify-program-engine.py --permuted-pages` with a freshly
prepared/bound bundle for the shared library hash in `artifacts.sha256`.

This is an S2 submilestone: it validates actual NPU cache access through shuffled
tables within one request's storage. Pools are not yet shared across concurrent
requests, recurrent slots remain single-request, and no multi-sequence decode is
claimed. The short fixtures cross BF16 page boundaries but do not cross a C8
32-token boundary. Longer boundary probes, batch metadata staging and shared
native state are the next gates. Packed/chunked prefill, radix prefix reuse and
long-context reference qualification remain outstanding.
