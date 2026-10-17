# Planned page-table capacities

Date: 2026-10-17. This milestone removes a fixed 32-entry binding ceiling;
it does **not** qualify 64K model execution. Serving admission remains 512 tokens.

The engine validates named and logical-state bindings against the registered
operator's compiled capacity before copying IDs or dispatching worker commands.
C8 tables use ceil(context/32) entries; index tables use ceil(context/64).
BF16 attention retains its actual 512-token, 16-token-page allocation until
window recycling is implemented. The physical recipe exposes the C8 table
shape, and the new contract header participates in backend revision hashing.

## Verification

- Local compiler token suite: 11 tests passed, including non-page-aligned
  context 129 table shapes (C8 5, index 3, BF16 32).
- Docker native ownership/contract tests passed through capacity 65,536:
  fragmented bindings, cross-owner rejection, failed-update rollback,
  reclamation, last-token addressing, count and rounding overflow rejection.
  These are metadata tests, not a 64K NPU decoder run.
- Full 46-layer TP8/EP8 NPU discovery, preparation and frozen execution each
  matched 99 standard fixture tokens with interleaved residents, permuted
  shared pages and packed prefill.
- Both binding APIs rejected 1,728 oversized requests (including UINT32_MAX)
  before worker dispatch. Valid retries and all 99 subsequent output tokens
  passed. See [binding results](binding-limits-results.json).
- vLLM Rust 0.25.1 HTTP checks passed full fixture text, SSE usage, six-request
  concurrency, queue isolation, cancellation/replacement, unsupported sampling,
  context overflow and shutdown. The mixed 504-token prompt had 5187.647 ms
  TTFT; its short peer completed in 677.591 ms. These are mixed-load measurements,
  not isolated TPOT. See [HTTP results](http-results.json).
- All eight ranks in each retained run reported frozen arenas, planned
  workspace, zero workspace growth, zero retained-step device allocations or
  internal completion boundaries, and zero activation host transfers.

Arena: 12,451,957,248 bytes/rank. Shared workspace: 130,916,352 bytes/rank.
Both remain unchanged; alignment absorbs the smaller C8 table allocation.
The HTTP run returned page occupancy to zero after 17 admissions.

## Artifact identity

Remote Docker: `inferfabric-92b` on `7.156.99.58`.
Working directory: `/data/p00603624/inferfabric/openpangu92`.
Artifacts use the `planned-page-tables-` prefix (`source`, `discovery`,
`workspace`, `preparation`, `bound`, `qualified`, `http`, `invalid-bindings`).
The final compiler revision-hash input addition was checked locally afterward;
it changes provenance hashing, not the tested executable instructions.

| Artifact | SHA256 |
|---|---|
| execution.bin | b594cde84eb73e477ce4e5789753103f4e70d0e33791b6094fbbd968a61e7cd8 |
| native library | 1883e75964d5310aaa821e15f3aee370ea72c46764d403591fd3475596fa26dc |
| Rust server | f964e6e2c77df8806656ebc70ec7baba5a1673b2a3b3094e2a60760313746d06 |

Compact rank audits retain source-report hashes. Raw reports remain in the
remote artifact directories. Standard fixture success does not resolve the
[previous reference boundary divergences](../2026-10-17-output-parity/README.md).
SWA recycling, independent pool budgeting, full learned-index decoder parity,
prefix state checkpoints and long-context latency/memory qualification remain open.
