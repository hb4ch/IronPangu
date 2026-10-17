# Native ring allocation and addressing integration

Date: 2026-10-17. Ring storage is now connected to the production compiler,
manifest, allocator adapter and native state. Full-model serving remains capped
at 512; this milestone does not qualify long-context decoder output.

The compiler derives BF16 cache retention from its attention consumer and the
compiled prefill width. Manifest states now carry the sliding retention policy;
the final bundle publishes these states alongside its invocation variants.
The viewer recipe distinguishes retained capacity from logical context and shows
the physical cache shape, window, chunk allowance and absolute block-table shape.
C8 and learned-index states retain their full histories.

The native contract independently derives the same capacity from the executable
consumer. Cache allocation uses that capacity; ownership bindings contain unique
physical pages, while the stable device table repeats those IDs across logical
positions. Append scatter uses the absolute position modulo retained capacity,
then the owned page mapping. RoPE and attention lengths remain absolute. Resident
selection, reset and release preserve the existing ownership and completion rules.
The Rust adapter now accepts this implemented BF16/page-16 layout and rejects
unsupported encodings or chunk allowances. Prefix checkpoint support stays false.

## Evidence

- Local compiler token suite: 13 tests; native-wrapper suite: 6 tests; state-pool
  suite: 8 tests. The compiler test checks all 30 BF16 cache states at context 129:
  each reserves 144 rows, exposes a nine-page physical shape and declares the
  four-token chunk allowance. Incompatible cache consumers are rejected.
- Docker native contract tests passed through 65,536 capacity. The compiled
  model passed all eight initialization bindings and 36 negative contracts.
- The NPU ring probe now instantiates the **production AttentionState class**:
  65,536 absolute append addresses agree with its expanded table; ownership uses
  34 reversed pages from the other initial resident partition. Cross-owner alias
  binding is rejected and release/reassignment preserves the table.
- **524 real scatter ACLGraph replays** write four-row BF16 chunks through that
  mapping across wraps and at positions through 65,535. Final device storage
  matches a CPU write mirror exactly, including untouched peer storage.
- The independent window-attention reference checks also pass again: 552,960
  values, maximum absolute error 0.00013493, and 120 graph replays bit-exact with
  eager outputs. Synthetic fixture uploads/output reads are test-only.
- Full 46-layer 92B TP8/EP8 discovery, workspace preparation and frozen execution
  each matched all 99 standard fixture tokens with interleaved residents,
  noncontiguous shared bindings and packed prefill.
- The rebuilt vLLM Rust frontend passed full HTTP fixture text, SSE/usage,
  six-request concurrency, queue isolation, peer-prefill cancellation/replacement,
  overflow/sampling rejection and graceful shutdown. The mixed 504-token prompt
  TTFT was **5276.810 ms**; the short peer finished in **704.757 ms**. These are
  mixed-load results, not isolated TPOT.

All eight native and HTTP audits report frozen arenas, planned workspace, no
workspace growth, no retained-step allocations/internal completion boundaries,
and no activation host transfers. Arena: 12,451,957,248 bytes/rank; workspace:
130,916,352 bytes/rank. HTTP page occupancy returned to zero after 17 admissions.
See [native results](native-results.json), [HTTP results](http-results.json),
[native audit](native-audit.json) and [HTTP audit](http-audit.json).

## Provenance and limits

Rust binaries were cross-compiled locally in WSL. Native compilation and hardware
tests ran in Docker `inferfabric-92b` on `7.156.99.58`. Remote working directory:
`/data/p00603624/inferfabric/openpangu92`; artifacts use prefix `native-ring-`
(`source`, `discovery`, `workspace`, `preparation`, `bound`, `qualified`, `http`).
The production-state/attention probe output is `native-ring-scatter.log`.

| Artifact | SHA256 |
|---|---|
| Native library | da110c1bb8bbb57ac3c907d0927c803be852a3867e141678ad4c53b1f2a66593 |
| Rust frontend | b54d31cd540c8fccfc98a0dc1c5e7c519a206c310ce63a78e7fd9ce3e877f51d |
| execution.bin | ae7c97fe4473488e0fd4829401d1d043ff3b2037ce1b71624aaab50c50576939 |
| sliding_window_probe | 4ed8b54bd737d475c5236f783fdfb97918f570958bb1a6374f4dc31229632cca |

At context 512, the ring is capped to the full-context allocation and does not
wrap. The full-model tests therefore establish regression compatibility, while
the separate production-state probe establishes wrap addressing/scatter behavior.
They do not establish full-model output after a wrap. Authoring, compiler and
native admission gates remain 512 pending longer-context decoder qualification
and unresolved reference-output differences. Prefix snapshots/COW and long-context
TTFT/TPOT/memory qualification remain open.
