# Shared native resident-state qualification

Date: 2026-10-17. Full 46-layer 92B, TP8/EP8, packed C8 on the DSL-selected
16 DSA layers and BF16 on the 30 SWA layers. Execution remained inside the assigned
NPU Docker container; Rust compilation was local.

`--resident-sequences 2` reserves two independent cache partitions and convolution
history slots in one native engine. The binary records this capacity, the viewer
shows expanded storage shapes, and the C ABI/Rust adapter select or reset one
resident without resetting its peer. Weights, activation scratch, workspace and
ACLGraph are shared. Per-resident page permutations survive switching; cross-partition
page bindings are rejected. Selection transfers control metadata only.

## Full-model results

- [Interleaved execution](interleaved-results.json): all 99 immutable fixture tokens
  match across nine arithmetic, geography and Chinese requests. The probe alternates
  residents at token boundaries, mixing prompt processing with decode and reusing
  finished slots while the peer remains live. Each rank/cache uses shuffled pages.
- [Serial execution](serial-results.json): the same 99 tokens match using the same
  two-resident binary. Average TPOT per request is 17.972–18.041 ms.
- [Page-boundary execution](boundaries-results.json): all 336 tokens match the
  previously qualified serial-layout oracle across 42 interleaved requests with
  shuffled pages and prompt lengths 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127,
  255, 480 and 504, each generating eight tokens. The [saved fixture](boundary-cases.json)
  uses repeated synthetic input IDs; this is isolation/layout equivalence, not an
  independent model-quality oracle. With a peer active, 504-token TTFT is about
  19.94–20.09 seconds: serial prompt processing still dominates.
- Interleaved average per-request TPOT is 38.16–46.89 ms while peers remain active;
  the final request averages 24.45 ms as its peer finishes. These are latency
  measurements including peer execution and admission/reset work, not kernel-only
  times or evidence of throughput scaling.

[Interleaved audits](interleaved-audit.json) and [serial audits](serial-audit.json)
verify all eight ranks: frozen arena 12,431,494,720 bytes/rank, fixed shared
workspace 130,916,352 bytes/rank, zero workspace growth, zero activation host
transfers, and zero allocations or internal completion boundaries in retained
token steps. Arena storage increases by 23,422,464 bytes/rank over the prior
single-resident control-bank plan, rather than duplicating the decoder. Device
HBM snapshots have device-wide scope and include runtime overhead.
[Boundary audits](boundaries-audit.json) pass the same all-rank gates.

## Identity and reproduction

SHA-256:

| Artifact | Digest |
|---|---|
| `execution.bin` | `10222d58dccc4ecb4aaef27cc936da765fc32273e9c24b80e61e0d6eef2083bf` |
| `libinferfabric_program.so` | `f34c218780b6e4959cded92749fe7fc84d319480686b460976d88c4c783e2549` |
| `program_engine_probe` | `d5a0a39d0022121babc0167961810bb3b014e14222ce5d18021d0ef3777f7977` |
| Immutable short-output fixture | `d543d78afd600b46215ed0a45fccede9a267cfec7f3170a5dc5c28421dd511a4` |

Compile with `--resident-sequences 2`; perform workspace discovery, fixed-workspace
preparation, then `bind-prepared-arena.py`. Run `qualify-program-engine.py` against
the bound bundle with `--interleave --permuted-pages` and then without either flag.
For the boundary run, use `--cases boundary-cases.json`; the harness repeats all
14 cases three times while reusing resident slots.
The harness checks full token output and all-rank residency/workspace evidence.
Raw preparation audits remain alongside the remote result directories; compact
audit files here record their hashes. Local compiler/IR/runtime tests and native
portable page-ownership tests cover capacity bounds and invalid mappings.

## Remaining scope

This is S2 resident storage and interleaving, **not packed multi-sequence execution**.
The HTTP worker remains serial until scheduler integration. Native context is
still capped at 512. Shared dynamic page allocation, SWA recycling, actual packed
prefill/decode, learned DSA, hybrid prefix restore and 64K reference/performance
qualification remain open. No activation offload or host fallback was introduced.
