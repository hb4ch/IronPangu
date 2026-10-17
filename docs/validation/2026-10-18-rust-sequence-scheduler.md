# Rust sequence submission and scheduler wiring

The Rust native engine now packs multi-request input for the native sequence
ABI. It validates member uniqueness and aggregate token capacity, reserves all
participating state leases in one submission ticket, and completes that ticket
only after the synchronous native call confirms all rank streams completed.
Errors poison shared state; completed arithmetic history advances per member.

The program worker gathers eligible residents into one native batch instead of
calling one forward per resident. It preserves output backpressure, completion-
ordered cancellation, prefix checkpoints, token budgets and round-robin
progress. A full 4096-token prefill quantum waits for sufficient budget instead
of being fragmented by a preceding decode. A batch may contain prefill and
decode when the combined real token count fits the budget.

Prefix compatibility is conservative: a batch history identifies sequence
geometry, peer token content and member ordinal, rather than asserting that
batching and serial arithmetic are interchangeable. This may reduce APC hits
until broader numerical equivalence is qualified; it avoids incorrect reuse.

Validation on local WSL:

- Native Rust library: nine tests passed, including descriptor layout, bounds,
  duplicate residents and prefix arithmetic identity.
- Frontend suite: 28 passed, one hardware test ignored; subsequently added
  cancellation and failure propagation cases passed in the five-test packed
  scheduler subset.
- Mixed member/output mapping and preserved 4K quantum tests passed.
- ARM64 frontend cross-compilation succeeded locally.
- Native capability accessor and engine rebuilt inside the remote Docker.

This is local scheduler/FFI validation, not an HTTP NPU serving qualification.
The native capacity accessor returns zero for an unqualified production handle;
only preparation handles currently expose batch capacity. Production batch
arena preparation and decode graphs must be completed before enabling the new
scheduler path in serving. Existing production behavior is retained meanwhile.
