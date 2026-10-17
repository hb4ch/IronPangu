# Explicit serving retention contract

Date: 2026-10-17. The state pool now distinguishes executable context from
physical retention. Native ring execution is **not enabled**; serving stays at 512.

`State.retention` is either `full` (the default when absent) or
`sliding_window { max_query_tokens }`. Attention `window_tokens` alone does not
authorize recycling. Sliding retention requires paged storage, a declared window,
enough capacity for every advertised chunk, and no checkpoint-support claim.
The capacity is checked against the same page-alignment formula qualified by the
[NPU ring probe](../2026-10-17-sliding-window/README.md).

The generic state pool reserves `ceil(min(request_tokens, retained_capacity)/page)`
unique owned pages for a ring. Absolute append positions map through that owned
ring; full-history components still reserve the full request. Executable context
and request token limits remain independently enforced. Retirement continues to
wait for every rank completion and outstanding borrower. Borrowing remains an
ownership operation, not an immutable prefix snapshot or permission to overwrite
pages read by another invocation.

The compiler still emits full retention and the native adapter explicitly rejects
ring manifests before loading the engine. This prevents a manifest edit from
silently enabling an addressing mode the executor does not yet implement. Missing
retention fields deserialize as full; full manifests retain their serialized shape.

## Verification

Local WSL tests passed: four serving-contract tests, eight state-pool tests,
six native-wrapper tests and twelve compiler token tests. Added coverage includes:

- Exact ring sizing, including 528 rows for single-token decode and 544 rows for
  four-token chunks at window 512/page 16; short-context capping and overflow.
- Missing window, undersized capacity, incompatible chunk and unsupported slot
  or checkpoint claims rejected; serialization round-trip/default compatibility.
- Two synthetic 64K requests on eight ranks, mixing a 544-row SWA ring with
  full-history C8 and recurrent state. Live window/chunk rows do not alias across
  page or ring boundaries through position 65,535, and peers remain disjoint.
- Atomic pressure rejection, all-rank/borrower retirement, stale-generation
  rejection and short-request page rounding. Window metadata without an explicit
  ring policy does not grant long-request admission.
- A structurally valid ring manifest is rejected by the current native adapter.

The Rust frontend was cross-compiled locally and tested in Docker against the
unchanged frozen `window-metadata-bound` TP8/EP8 native plan. Full fixture text,
SSE/usage agreement, six concurrent requests, queue isolation, peer-prefill
cancellation/replacement, overflow and unsupported-sampling rejection, and
graceful shutdown passed. See [HTTP results](http-results.json).

Mixed workload: 504-token TTFT **5252.930 ms**, short-peer completion **693.743 ms**.
These are not isolated TPOT measurements. Seventeen admissions returned occupied
pool bytes to zero. All eight [rank audits](http-audit.json) retain frozen arenas,
planned workspace, zero growth/step allocations/internal completion boundaries,
and zero activation host transfers. Arena and workspace remain 12,451,957,248 and
130,916,352 bytes per rank, respectively.

Remote evidence: `/data/p00603624/inferfabric/openpangu92/retention-contract-http`,
Docker `inferfabric-92b`, host `7.156.99.58`.

| Artifact | SHA256 |
|---|---|
| Rust frontend | 4f31d83c489117ecd256db7f3583f2cd3db0d91818e52425a7342416fce891b9 |
| Native library | da34f18deea69dae4024a712043a49dd60c456939a5125b67f3a3146f6361790 |
| execution.bin | ae7c97fe4473488e0fd4829401d1d043ff3b2037ce1b71624aaab50c50576939 |

Next integration must emit the policy from the plan, allocate native ring storage,
expand unique ownership bindings into absolute device tables, and scatter through
that mapping. Only then can the adapter guard be removed, followed by full-model
wrap/reference qualification. These ownership tests do not establish 64K model
correctness or performance, and prefix checkpoint/COW work remains outstanding.
